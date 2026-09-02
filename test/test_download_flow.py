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
    assert task["model_type"] == "lora"
    assert task["error"] == DISABLED


# --- a file this library cannot hold -----------------------------------------

GGUF_METADATA = {
    "model_id": "2179031",
    "version_id": "2453732",
    "model_dict": {"id": 2179031, "name": "Z-Image-Turbo-GGUF", "type": "Checkpoint"},
    "version_dict": {
        "id": 2453732,
        "downloadUrl": "https://civitai.com/api/download/models/2453732",
        "files": [{
            "primary": True,
            "name": "zImageTurboGGUF_q80.gguf",
            "sizeKB": 7055378,
        }],
    },
}


def test_a_gguf_version_is_refused_before_it_is_downloaded(civitdl):
    """Civitai names the file in the metadata, so 7GB need not arrive first."""
    with patch.object(utils, "get_safe_metadata", return_value=GGUF_METADATA), \
            patch.object(utils, "find_model_files", return_value=[]):
        with pytest.raises(HTTPException) as excinfo:
            utils._civitdl(2179031, 2453732, "key")

    assert excinfo.value.status_code == 501
    assert "zImageTurboGGUF_q80.gguf" in excinfo.value.detail
    assert "API Key" not in excinfo.value.detail
    assert civitdl.call_count == 0


def test_the_async_task_refuses_a_gguf_without_downloading(civitdl):
    with patch.object(utils, "get_safe_metadata", return_value=GGUF_METADATA), \
            patch.object(utils, "find_model_files", return_value=[]):
        task_id = utils.create_task(2179031, 2453732)
        utils._civitdl_async_worker(task_id, 2179031, 2453732, "key")

    task = utils.get_task(task_id)
    assert task["status"] == "failed"
    assert task["model_name"] == "Z-Image-Turbo-GGUF"
    assert "zImageTurboGGUF_q80.gguf" in task["error"]
    assert "API Key" not in task["error"]
    assert civitdl.call_count == 0


def test_a_file_that_lands_anyway_is_named_rather_than_blamed(civitdl, tmp_path):
    """Backstop: whatever the metadata claimed, report what actually arrived."""
    def wrote_a_gguf(*args, **kwargs):
        model_dir = tmp_path / "Heime-mid_2805786-vid_3163627"
        extra_data = model_dir / "extra_data-vid_3163627"
        extra_data.mkdir(parents=True)
        (extra_data / "model_dict-mid_2805786-vid_3163627.json").write_text("{}")
        (model_dir / "heime_v10-mid_2805786-vid_3163627.gguf").write_bytes(b"w")

    civitdl.side_effect = wrote_a_gguf

    with patch.object(utils, "MODEL_ROOT_PATH", str(tmp_path)), \
            patch.dict(utils.MODEL_TYPE_TO_FOLDER, {"lora": str(tmp_path)}), \
            patch.object(utils, "get_safe_metadata", return_value=METADATA), \
            patch("requests.Session.request", return_value=MagicMock(status_code=200)):
        with pytest.raises(HTTPException) as excinfo:
            utils._civitdl(2805786, 3163627, "key")

    assert excinfo.value.status_code == 501
    assert "heime_v10-mid_2805786-vid_3163627.gguf" in excinfo.value.detail
    # The metadata civitdl writes alongside it is not a model file.
    assert ".json" not in excinfo.value.detail


def test_a_download_that_wrote_nothing_says_that(civitdl):
    """No refusal, no timeout, no file: none of that is a missing API key."""
    with patch.object(utils, "get_safe_metadata", return_value=METADATA), \
            patch.object(utils, "find_model_files", return_value=[]), \
            patch("requests.Session.request", return_value=MagicMock(status_code=200)):
        with pytest.raises(HTTPException) as excinfo:
            utils._civitdl(2805786, 3163627, "key")

    assert excinfo.value.status_code == 502
    assert "without writing a model file" in excinfo.value.detail
    assert "API Key" not in excinfo.value.detail
