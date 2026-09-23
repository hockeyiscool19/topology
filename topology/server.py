"""HTTP API + static frontend. Run with `topology serve`."""

from __future__ import annotations

import shutil
import tempfile
import threading
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from topology import boundary
from topology.config import JobSpec
from topology.pipeline import JobResult, run_job

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
MAX_JOBS = 24


@dataclass
class Job:
    id: str
    spec: JobSpec
    status: str = "queued"  # queued | running | done | error | cancelled
    message: str = "Queued"
    progress: float = 0.0
    error: Optional[str] = None
    result: Optional[JobResult] = None
    preview: Optional[dict[str, Any]] = None
    zip_path: Optional[Path] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def public(self) -> dict[str, Any]:
        d = {"id": self.id, "status": self.status, "message": self.message,
             "progress": round(self.progress, 3), "error": self.error}
        if self.status == "done":
            d["result"] = self.preview
        return d


class JobStore:
    def __init__(self) -> None:
        self.jobs: "OrderedDict[str, Job]" = OrderedDict()
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.tmp = Path(tempfile.mkdtemp(prefix="topology-"))

    def submit(self, spec: JobSpec) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], spec=spec)
        with self.lock:
            self.jobs[job.id] = job
            while len(self.jobs) > MAX_JOBS:
                _, old = self.jobs.popitem(last=False)
                if old.zip_path:
                    shutil.rmtree(old.zip_path.parent, ignore_errors=True)
        self.pool.submit(self._run, job)
        return job

    def get(self, job_id: str) -> Job:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        return job

    def _run(self, job: Job) -> None:
        def progress(msg: str, frac: float) -> None:
            job.message, job.progress = msg, frac

        with job.lock:
            if job.status == "cancelled":
                return
            job.status = "running"
        try:
            job.result = run_job(job.spec, progress=progress)
            job.preview = job.result.preview()
            job.status = "done"
        except Exception as e:  # surfaced to the UI
            traceback.print_exc()
            job.error = str(e) or e.__class__.__name__
            job.status = "error"
            job.message = "Failed"

    def cancel(self, job: Job) -> None:
        """Cancelling only skips queued work; a running job finishes but nobody waits for it."""
        with job.lock:
            if job.status == "queued":
                job.status, job.message = "cancelled", "Cancelled"

    def zip_for(self, job: Job) -> Path:
        with job.lock:
            if job.zip_path is None or not job.zip_path.exists():
                if job.result is None:
                    raise HTTPException(409, "Job has no result yet")
                job.zip_path = job.result.write(self.tmp / job.id)
        return job.zip_path


def create_app() -> FastAPI:
    app = FastAPI(title="Topology", version="1.0")
    app.add_middleware(GZipMiddleware, minimum_size=2048)
    store = JobStore()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @lru_cache(maxsize=1)
    def _states() -> dict[str, Any]:
        return boundary.states_geojson()

    # Warm the boundary cache in the background so the map's first request is fast.
    threading.Thread(target=_states, daemon=True).start()

    @app.get("/api/states")
    def states() -> JSONResponse:
        return JSONResponse(_states(), headers={"Cache-Control": "public, max-age=86400"})

    @app.post("/api/jobs", status_code=202)
    def create_job(body: dict[str, Any]) -> dict[str, Any]:
        try:
            spec = JobSpec.from_dict(body)
            spec.validate()
            if spec.geojson is not None:
                boundary.geometry_from_geojson(spec.geojson)
        except (TypeError, ValueError, KeyError) as e:
            raise HTTPException(422, str(e))
        return store.submit(spec).public()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return store.get(job_id).public()

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def cancel_job(job_id: str) -> None:
        store.cancel(store.get(job_id))

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str) -> FileResponse:
        job = store.get(job_id)
        z = store.zip_for(job)
        return FileResponse(z, media_type="application/zip", filename=z.name)

    if WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")

    return app
