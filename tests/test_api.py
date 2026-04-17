"""
HTTP-layer tests for the FastAPI application.

All tests are fast: worker.start/stop are mocked so no threads or watchdog
observers are spawned, and the file system is not touched by default.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from main import app
from proofreader import worker
from proofreader.worker import _jobs, _jobs_lock

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """TestClient with worker lifecycle mocked out."""
    with patch("main.worker.start"), patch("main.worker.stop"):
        with TestClient(app) as c:
            yield c


@pytest.fixture(autouse=True)
def clear_jobs():
    """Reset global job state between tests."""
    with _jobs_lock:
        _jobs.clear()
    yield
    with _jobs_lock:
        _jobs.clear()


# ---------------------------------------------------------------------------
# DELETE /results/{job_id}
# ---------------------------------------------------------------------------


def test_delete_complete_job_returns_200(client: TestClient) -> None:
    worker._set_job("aabbcc", status="complete", verdict="PASS")
    with patch("main.worker.delete_job", return_value=True) as mock_del:
        resp = client.delete("/results/aabbcc")
    assert resp.status_code == 200
    assert resp.json() == {"status": "deleted"}
    mock_del.assert_called_once_with("aabbcc")


def test_delete_error_job_returns_200(client: TestClient) -> None:
    worker._set_job("err001", status="error", verdict="ERROR")
    with patch("main.worker.delete_job", return_value=True):
        resp = client.delete("/results/err001")
    assert resp.status_code == 200


def test_delete_unknown_job_returns_404(client: TestClient) -> None:
    resp = client.delete("/results/doesnotexist")
    assert resp.status_code == 404


def test_delete_queued_job_returns_409(client: TestClient) -> None:
    worker._set_job("q001", status="queued")
    resp = client.delete("/results/q001")
    assert resp.status_code == 409


def test_delete_processing_job_returns_409(client: TestClient) -> None:
    worker._set_job("p001", status="processing")
    resp = client.delete("/results/p001")
    assert resp.status_code == 409


def test_delete_does_not_call_delete_job_for_non_terminal(client: TestClient) -> None:
    """delete_job() must not be called when the job is not in a terminal state."""
    worker._set_job("q002", status="queued")
    with patch("main.worker.delete_job") as mock_del:
        client.delete("/results/q002")
    mock_del.assert_not_called()


# ---------------------------------------------------------------------------
# POST /results/{job_id}/requeue
# ---------------------------------------------------------------------------


def test_requeue_complete_job_returns_200(client: TestClient) -> None:
    worker._set_job("rqa001", status="complete", verdict="PASS")
    with patch("main.worker.requeue_job", return_value=True) as mock_rq:
        resp = client.post("/results/rqa001/requeue")
    assert resp.status_code == 200
    assert resp.json() == {"status": "queued", "job_id": "rqa001"}
    mock_rq.assert_called_once_with("rqa001")


def test_requeue_error_job_returns_200(client: TestClient) -> None:
    worker._set_job("rqa002", status="error", verdict="ERROR")
    with patch("main.worker.requeue_job", return_value=True):
        resp = client.post("/results/rqa002/requeue")
    assert resp.status_code == 200


def test_requeue_unknown_job_returns_404(client: TestClient) -> None:
    resp = client.post("/results/doesnotexist/requeue")
    assert resp.status_code == 404


def test_requeue_queued_job_returns_409(client: TestClient) -> None:
    worker._set_job("rqa003", status="queued")
    resp = client.post("/results/rqa003/requeue")
    assert resp.status_code == 409


def test_requeue_processing_job_returns_409(client: TestClient) -> None:
    worker._set_job("rqa004", status="processing")
    resp = client.post("/results/rqa004/requeue")
    assert resp.status_code == 409


def test_requeue_returns_500_when_requeue_job_fails(client: TestClient) -> None:
    """requeue_job() returning False (e.g. original PDF missing) surfaces as 500."""
    worker._set_job("rqa005", status="complete", verdict="PASS")
    with patch("main.worker.requeue_job", return_value=False):
        resp = client.post("/results/rqa005/requeue")
    assert resp.status_code == 500
