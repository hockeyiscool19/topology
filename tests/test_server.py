import time
import zipfile
import io

from fastapi.testclient import TestClient

from topology import pipeline
from topology.server import create_app
from tests.test_pipeline import SQUARE, Synthetic


def test_job_lifecycle(monkeypatch):
    monkeypatch.setattr(pipeline, "TerrariumSource", Synthetic)
    c = TestClient(create_app())
    assert c.post("/api/jobs", json={"state": "VT", "geojson": SQUARE}).status_code == 422
    assert c.post("/api/jobs", json={"geojson": SQUARE, "layers": {"count": 0}}).status_code == 422
    assert c.post("/api/jobs", json={"geojson": SQUARE, "bogus": 1}).status_code == 422

    r = c.post("/api/jobs", json={"geojson": SQUARE, "name": "sq", "board": {"resolution_in": 0.03}, "layers": {"count": 5}})
    assert r.status_code == 202
    jid = r.json()["id"]
    for _ in range(200):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert j["status"] == "done", j
    res = j["result"]
    assert len(res["stats"]["layers"]) == 5 and len(res["svg"]["layers"]) == 5
    hm = res["heightmap"]
    assert hm["width"] > 0 and hm["height"] > 0

    d = c.get(f"/api/jobs/{jid}/download")
    assert d.status_code == 200
    assert "manifest.json" in zipfile.ZipFile(io.BytesIO(d.content)).namelist()
    assert c.get("/api/jobs/nope").status_code == 404


def test_cancel_queued_job(monkeypatch):
    import threading
    gate = threading.Event()

    class Slow(Synthetic):
        def sample(self, layout):
            gate.wait(5)
            return super().sample(layout)

    monkeypatch.setattr(pipeline, "TerrariumSource", Slow)
    c = TestClient(create_app())
    body = {"geojson": SQUARE, "board": {"resolution_in": 0.05}, "layers": {"count": 3}}
    ids = [c.post("/api/jobs", json=body).json()["id"] for _ in range(3)]  # 2 workers -> third is queued
    assert c.delete(f"/api/jobs/{ids[2]}").status_code == 204
    gate.set()
    for _ in range(200):
        states = [c.get(f"/api/jobs/{i}").json()["status"] for i in ids]
        if states[0] == states[1] == "done":
            break
        time.sleep(0.05)
    assert states == ["done", "done", "cancelled"]
