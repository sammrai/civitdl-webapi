"""Download failures: what gets reported, and what gets downloaded twice."""

import threading
import time

from unittest.mock import MagicMock, patch

import pytest

from fastapi import HTTPException

from helpers.core.utils import APIException

from app import utils

DOWNLOAD_URL = "https://civitai.com/api/download/models/3163627"
DISABLED = "The creator of this asset has disabled downloads on this file"

METADATA = {
    "model_id": "2805786",
    "version_id": "3163627",
    "model_dict": {"id": 2805786, "name": "Heime", "type": "LORA"},
    "version_dict": {"id": 3163627, "downloadUrl": DOWNLOAD_URL},
}


def response(status, payload=None):
    res = MagicMock()
    res.status_code = status
    res.json.return_value = payload if payload is not None else {}
    return res


def refusal():
    return response(401, {"error": "Unauthorized", "message": DISABLED})


def get_returning(res):
    """Patch the underlying requests.Session.get so only our wrapper runs."""
    return patch("requests.Session.get", return_value=res)


@pytest.fixture
def civitdl():
    """Stub out civitdl's argument parsing and downloader."""
    with patch.object(utils, "wrap_cli_args", return_value={"source_strings": [], "rootdir": "/data"}), \
            patch.object(utils, "BatchOptions", MagicMock()), \
            patch.object(utils, "batch_download") as batch_download:
        yield batch_download


# --- the reason a download failed -------------------------------------------

def test_refusal_message_is_kept():
    session = utils._CivitaiSession()
    with get_returning(refusal()):
        session.get(DOWNLOAD_URL)

    assert session.refusal == DISABLED


def test_successful_download_records_nothing():
    session = utils._CivitaiSession()
    with get_returning(response(200)):
        session.get(DOWNLOAD_URL)

    assert session.refusal is None


def test_refusal_elsewhere_is_ignored():
    session = utils._CivitaiSession()
    with get_returning(response(401, {"message": "nope"})):
        session.get("https://civitai.com/api/v1/models/2805786")

    assert session.refusal is None


def test_non_json_refusal_is_survivable():
    res = response(401)
    res.json.side_effect = ValueError("not json")
    session = utils._CivitaiSession()
    with get_returning(res):
        session.get(DOWNLOAD_URL)

    assert session.refusal is None


def test_rate_limiting_is_not_reported_as_not_found():
    with patch.object(utils, "find_model_files", return_value=[]), \
            patch.object(utils, "get_safe_metadata", side_effect=APIException(429, "rate limited")):
        with pytest.raises(HTTPException) as excinfo:
            utils._civitdl(28205, 47670)

    assert excinfo.value.status_code == 429
    assert excinfo.value.detail == utils.RATE_LIMITED


def test_a_missing_model_is_still_a_404():
    with patch.object(utils, "find_model_files", return_value=[]), \
            patch.object(utils, "get_safe_metadata", side_effect=APIException(404, "nope")):
        with pytest.raises(HTTPException) as excinfo:
            utils._civitdl(999999999)

    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "Model not found on Civitai."


# --- duplicate requests ------------------------------------------------------

def test_concurrent_requests_for_one_version_download_once(civitdl):
    """Three simultaneous requests must run civitdl once, not three times."""
    found = []

    def download(*args, **kwargs):
        time.sleep(0.3)          # keep the window open for the other two
        found.append(MagicMock())

    civitdl.side_effect = download

    with patch.object(utils, "get_safe_metadata", return_value=METADATA), \
            patch.object(utils, "find_model_files", side_effect=lambda *a, **k: list(found)):
        threads = [
            threading.Thread(target=utils._civitdl, args=(2805786, 3163627, "key"))
            for _ in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert civitdl.call_count == 1


# --- what a failed task says -------------------------------------------------

def test_the_task_records_which_model_it_was(civitdl):
    def refused_download(*args, **kwargs):
        # civitdl asks the download endpoint through the session we handed it.
        kwargs["batchOptions"].session.get(DOWNLOAD_URL)

    civitdl.side_effect = refused_download

    with patch.object(utils, "get_safe_metadata", return_value=METADATA), \
            patch.object(utils, "find_model_files", return_value=[]), \
            get_returning(refusal()):
        task_id = utils.create_task(2805786, 3163627)
        utils._civitdl_async_worker(task_id, 2805786, 3163627, "key")

    task = utils.get_task(task_id)
    assert task["status"] == "failed"
    assert task["model_name"] == "Heime"
    assert task["model_type"] == "LORA"
    assert task["error"] == DISABLED
