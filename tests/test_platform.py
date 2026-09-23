"""Platform contract: the same app answers at "/" (topology-cnc.web.app, Cloud Run URL) and under /topology
(eisensoftware.web.app Hosting rewrite, which forwards the full path)."""

import pytest
from fastapi.testclient import TestClient
from starlette.responses import RedirectResponse

from topology import boundary, server
from topology.server import PrefixMiddleware, create_app

STATES = {"type": "FeatureCollection", "features": []}
CLOUD_RUN_HOST = "topology-382031913173.us-central1.run.app"


@pytest.fixture
def client(monkeypatch, tmp_path):
    """App with a stand-in frontend build and no network (the states warm-up thread calls the stub)."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text('<script type="module" src="./assets/app.js"></script>')
    (tmp_path / "assets" / "app.js").write_text('fetch("api/states")')
    monkeypatch.setattr(server, "WEB_DIST", tmp_path)
    monkeypatch.setattr(boundary, "states_geojson", lambda: STATES)
    return TestClient(create_app(), follow_redirects=False)


@pytest.mark.parametrize("base", ["", "/topology"])
def test_frontend_and_api_under_both_bases(client, base):
    r = client.get(f"{base}/")
    assert r.status_code == 200 and "./assets/app.js" in r.text
    r = client.get(f"{base}/assets/app.js")
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]
    r = client.get(f"{base}/api/states")
    assert r.status_code == 200 and r.json() == STATES
    assert client.post(f"{base}/api/jobs", json={"bogus": 1}).status_code == 422
    assert client.get(f"{base}/api/jobs/nope").status_code == 404
    assert client.get(f"{base}/assets/missing.js").status_code == 404


def test_prefix_is_a_whole_path_segment(client):
    assert client.get("/topologyx/assets/app.js").status_code == 404
    assert client.get("/topology/topology/").status_code == 404


def test_prefix_without_slash_redirects_relative(client):
    r = client.get("/topology")
    assert r.status_code == 307 and r.headers["location"] == "topology/"
    r = client.get("/topology?state=VT", headers={"host": CLOUD_RUN_HOST})
    assert r.status_code == 307 and r.headers["location"] == "topology/?state=VT"
    r = client.get("/topology", follow_redirects=True)
    assert r.status_code == 200 and str(r.url) == "http://testserver/topology/"


@pytest.mark.parametrize("base", ["", "/topology"])
def test_starlette_redirects_become_relative(monkeypatch, base):
    """Without a frontend mount, Starlette's trailing-slash redirect is absolute (Host header, http:)."""
    monkeypatch.setattr(server, "WEB_DIST", server.WEB_DIST / "does-not-exist")
    monkeypatch.setattr(boundary, "states_geojson", lambda: STATES)
    c = TestClient(create_app(), follow_redirects=False)
    r = c.get(f"{base}/api/health/?x=1", headers={"host": CLOUD_RUN_HOST})
    assert r.status_code == 307 and r.headers["location"] == f"{base}/api/health?x=1"


def test_external_redirects_untouched():
    async def app(scope, receive, send):
        await RedirectResponse("https://example.com/elsewhere")(scope, receive, send)

    r = TestClient(PrefixMiddleware(app), follow_redirects=False).get("/topology/x")
    assert r.headers["location"] == "https://example.com/elsewhere"
