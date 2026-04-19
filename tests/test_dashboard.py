"""Smoke tests for the control UI."""

from fastapi.testclient import TestClient

from middleware.app import app


def test_dashboard_root_serves_html() -> None:
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "AI Router" in r.text


def test_dashboard_static_css() -> None:
    client = TestClient(app)
    r = client.get("/static/dashboard/dashboard.css")
    assert r.status_code == 200
    assert "dashboard.css" in r.headers.get("content-type", "") or "css" in r.headers.get(
        "content-type", ""
    )
