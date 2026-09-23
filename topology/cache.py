"""On-disk cache and HTTP helpers."""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = "topology-cnc/1.0 (+https://github.com/hockeyiscool19/topology)"


def cache_dir(*parts: str) -> Path:
    root = Path(os.environ.get("TOPOLOGY_CACHE", Path.home() / ".cache" / "topology"))
    p = root.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def fetch_bytes(url: str, retries: int = 4, timeout: float = 30.0) -> bytes:
    """GET with exponential backoff. 4xx errors other than 429 are not retried."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if 400 <= e.code < 500 and e.code != 429:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(min(8.0, 0.5 * 2**attempt))
    raise RuntimeError(f"Failed to download {url}: {last}")


def cached_download(url: str, path: Path) -> Path:
    """Download url to path once; writes atomically so partial files never poison the cache."""
    if path.exists() and path.stat().st_size > 0:
        return path
    data = fetch_bytes(url, timeout=120.0)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path
