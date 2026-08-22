"""The async download endpoints, end to end through HTTP.

These three endpoints had no HTTP-level test at all, which is where every bug
found so far has been: /status returning 404, duplicate downloads, and a failed
task that did not say which model it was.
"""

import re
import time

from unittest.mock import MagicMock, patch

import pytest

from fastapi.testclient import TestClient

from app import utils
from app.main import app
from app.models import ModelInfo

METADATA = {
    "model_id": "16014",
    "version_id": "28907",
    "model_dict": {
        "id": 16014,
        "name": "Anime Lineart",
        "type": "LORA",
        "modelVersions": [{"id": 28907, "files": [{"primary": True, "sizeKB": 18541}]}],
    },
    "version_dict": {
        "id": 28907,
        "downloadUrl": "https://civitai.com/api/download/models/28907",
    },
}

DOWNLOADED = ModelInfo(
    model_id=16014,
    version_id=28907,
    model_dir="/data/models/Lora/Anime Lineart-mid_16014-vid_28907",
    filename="animeoutlineV4_16-mid_16014-vid_28907.safetensors",
    model_type="lora",
    name="Anime Lineart",
    description="",
    created_at="2026-01-01T00:00:00Z",
)


@pytest.fixture
def client():
    return TestClient(app)


def wait_for(client, task_id, timeout=10):
    """Poll a task until it reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/status/{task_id}").json()
        if body["status"] in ("finished", "failed"):
            return body
        time.sleep(0.05)
    pytest.fail(f"task stayed {body['status']}")


def stub_civitdl(writes=None):
    """Patch out civitdl. `writes` is what find_model_files reports afterwards."""
    found = []

    def download(*args, **kwargs):
        if writes:
            found.extend(writes)

    return patch.multiple(
        utils,
        wrap_cli_args=MagicMock(return_value={"source_strings": [], "rootdir": "/data"}),
        BatchOptions=MagicMock(),
        batch_download=MagicMock(side_effect=download),
        get_safe_metadata=MagicMock(return_value=METADATA),
        find_model_files=MagicMock(side_effect=lambda *a, **k: list(found)),
    )


def test_a_download_can_be_started_and_polled(client):
    with stub_civitdl(writes=[DOWNLOADED]):
        started = client.post("/models/16014/versions/28907/async")
        assert started.status_code == 200

        task_id = started.json()["task_id"]
        assert started.json()["status_url"] == f"/status/{task_id}"

        task = wait_for(client, task_id)

    assert task["status"] == "finished"
    assert task["progress"] == 100
    assert task["result"]["filename"] == DOWNLOADED.filename
    assert task["error"] is None


def test_the_task_says_which_model_it_is(client):
    """A caller polling /status must be able to tell what it asked for."""
    with stub_civitdl(writes=[DOWNLOADED]):
        task_id = client.post("/models/16014/versions/28907/async").json()["task_id"]
        task = wait_for(client, task_id)

    assert task["model_id"] == 16014
    assert task["version_id"] == 28907
    assert task["model_name"] == "Anime Lineart"
    assert task["model_type"] == "LORA"


def test_a_download_that_writes_nothing_fails_with_a_reason(client):
    with stub_civitdl(writes=None), \
            patch("requests.Session.get", return_value=MagicMock(status_code=200)):
        task_id = client.post("/models/16014/async").json()["task_id"]
        task = wait_for(client, task_id)

    assert task["status"] == "failed"
    assert task["result"] is None
    assert task["error"] == utils.DOWNLOAD_REFUSED


def test_polling_an_unknown_task_is_404(client):
    assert client.get("/status/does-not-exist").status_code == 404


def test_an_already_downloaded_model_finishes_without_downloading(client):
    # Present before the request: no civitdl run should happen.
    with patch.object(utils, "find_model_files", return_value=[DOWNLOADED]), \
            patch.object(utils, "get_safe_metadata", return_value=METADATA), \
            patch.object(utils, "batch_download") as batch_download:
        task_id = client.post("/models/16014/versions/28907/async").json()["task_id"]
        task = wait_for(client, task_id)

        assert batch_download.call_count == 0

    assert task["status"] == "finished"
    assert task["progress"] == 100


def test_the_image_ships_a_single_uvicorn_worker():
    """Task state is a process-local dict, so extra workers break /status.

    Nothing else can catch that: it needs real processes, and TestClient runs
    in one. This guards the decision until the state moves out of process.
    """
    dockerfile = open("/app/Dockerfile", encoding="utf-8").read()
    workers = re.search(r'"--workers",\s*"(\d+)"', dockerfile)

    assert workers, "the CMD no longer sets --workers"
    assert workers.group(1) == "1", (
        "Raising the worker count needs task state shared between processes; "
        "see 'Single worker, on purpose' in CLAUDE.md"
    )
