"""Downloads that never end, and the wreckage they leave behind.

A 12GB checkpoint sat at "downloading" for an hour and a half with nothing on
disk, no error and no civitdl log line, and every later request for the same
version did exactly the same: civitdl passes no timeout to requests, and the
per-version lock a hung download holds is waited on forever.
"""

import threading

from unittest.mock import MagicMock, patch

import pytest
import requests

from fastapi import HTTPException

from app import utils

DOWNLOAD_URL = "https://civitai.com/api/download/models/3163627"

METADATA = {
    "model_id": "2805786",
    "version_id": "3163627",
    "model_dict": {"id": 2805786, "name": "Heime", "type": "LORA"},
    "version_dict": {"id": 3163627, "downloadUrl": DOWNLOAD_URL},
}


@pytest.fixture
def civitdl():
    """Stub out civitdl's argument parsing and downloader."""
    with patch.object(utils, "wrap_cli_args", return_value={"source_strings": [], "rootdir": "/data"}), \
            patch.object(utils, "BatchOptions", MagicMock()), \
            patch.object(utils, "batch_download") as batch_download:
        yield batch_download


# --- requests that can end ---------------------------------------------------

def test_every_civitai_request_carries_a_timeout():
    """civitdl asks through this session and never passes one itself."""
    with patch("requests.Session.request", return_value=MagicMock(status_code=200)) as request:
        utils._CivitaiSession().get(DOWNLOAD_URL, stream=True)

    assert request.call_args.kwargs["timeout"] == utils.CIVITAI_TIMEOUT


def test_a_caller_that_wants_its_own_timeout_keeps_it():
    with patch("requests.Session.request", return_value=MagicMock(status_code=200)) as request:
        utils._CivitaiSession().get(DOWNLOAD_URL, timeout=5)

    assert request.call_args.kwargs["timeout"] == 5


def test_a_download_that_timed_out_does_not_blame_the_api_key(civitdl):
    """civitdl swallows the timeout, so the session has to remember it."""
    def timed_out(*args, **kwargs):
        try:
            kwargs["batchOptions"].session.get(DOWNLOAD_URL, stream=True)
        except requests.RequestException:
            pass          # what civitdl's retry loop does with it

    civitdl.side_effect = timed_out

    task_id = utils.create_task(2805786, 3163627)
    with patch.object(utils, "get_safe_metadata", return_value=METADATA), \
            patch.object(utils, "find_model_files", return_value=[]), \
            patch("requests.Session.request", side_effect=requests.exceptions.ReadTimeout("read timed out")):
        utils._civitdl_async_worker(task_id, 2805786, 3163627, "key")

    task = utils.get_task(task_id)
    assert task["status"] == "failed"
    assert "API Key" not in task["error"]
    assert "ReadTimeout" in task["error"]


def test_a_body_that_stops_arriving_is_remembered_too():
    """A download stalls mid-file; that raises out of iter_content, not get()."""
    response = MagicMock()
    response.iter_content.side_effect = requests.exceptions.ConnectionError("read timed out")

    session = utils._CivitaiSession()
    with patch("requests.Session.request", return_value=response):
        streaming = session.get(DOWNLOAD_URL, stream=True)
        with pytest.raises(requests.RequestException):
            list(streaming.iter_content(1024))

    assert "ConnectionError" in session.transport_error


# --- a hung download must not wedge the version ------------------------------

def test_a_hung_download_does_not_wedge_later_requests(civitdl):
    """The lock outlives the request that took it, so waiting on it must end."""
    stuck = utils._download_lock(2805786, 3163627)
    assert stuck.acquire(), "lock left held by an earlier test"

    task_id = utils.create_task(2805786, 3163627)
    try:
        with patch.object(utils, "DOWNLOAD_LOCK_TIMEOUT", 0.2), \
                patch.object(utils, "get_safe_metadata", return_value=METADATA), \
                patch.object(utils, "find_model_files", return_value=[]):
            worker = threading.Thread(
                target=utils._civitdl_async_worker,
                args=(task_id, 2805786, 3163627, "key"),
                daemon=True,   # it never returns without the fix
            )
            worker.start()
            worker.join(timeout=10)

            assert not worker.is_alive(), "the request blocked on the hung download"
    finally:
        stuck.release()

    task = utils.get_task(task_id)
    assert task["status"] == "failed"
    assert "stuck" in task["error"]
    assert civitdl.call_count == 0


def test_a_hung_download_does_not_wedge_the_synchronous_endpoint():
    stuck = utils._download_lock(2805786, 3163628)
    assert stuck.acquire(), "lock left held by an earlier test"

    metadata = dict(METADATA, version_id="3163628")
    try:
        with patch.object(utils, "DOWNLOAD_LOCK_TIMEOUT", 0.2), \
                patch.object(utils, "wrap_cli_args", return_value={"source_strings": [], "rootdir": "/data"}), \
                patch.object(utils, "get_safe_metadata", return_value=metadata), \
                patch.object(utils, "find_model_files", return_value=[]):
            with pytest.raises(HTTPException) as excinfo:
                utils._civitdl(2805786, 3163628, "key")
    finally:
        stuck.release()

    assert excinfo.value.status_code == 503


# --- the size the disk check and the progress bar use ------------------------

def test_the_expected_size_is_the_requested_versions():
    """modelVersions[0] is the newest version, not the one that was asked for."""
    metadata = {
        "model_id": "2168935",
        "version_id": "2442439",
        "model_dict": {
            "id": 2168935,
            "modelVersions": [
                {"id": 9999999, "files": [{"primary": True, "sizeKB": 6 * 1024 * 1024}]},
                {"id": 2442439, "files": [{"primary": True, "sizeKB": 12 * 1024 * 1024}]},
            ],
        },
        "version_dict": {"id": 2442439},
    }

    assert utils._primary_file_size(metadata) == 12 * 1024 * 1024 * 1024


def test_the_expected_size_prefers_the_primary_file():
    """A version can ship a VAE next to the model; civitdl fetches the primary."""
    metadata = {
        "version_dict": {
            "id": 2442439,
            "files": [
                {"primary": False, "sizeKB": 310 * 1024},
                {"primary": True, "sizeKB": 11 * 1024 * 1024},
            ],
        },
    }

    assert utils._primary_file_size(metadata) == 11 * 1024 * 1024 * 1024


# --- what a failed download leaves on disk -----------------------------------

def partial_download(root, name="Heime-mid_2805786-vid_3163627"):
    """What civitdl writes before it gets to the model file."""
    extra_data = root / name / "extra_data-vid_3163627"
    extra_data.mkdir(parents=True)
    (extra_data / "model_dict-mid_2805786-vid_3163627.json").write_text(
        '{"type": "LORA", "name": "Heime"}')
    (extra_data / "137756956.jpeg").write_bytes(b"\xff\xd8\xff")
    return root / name


def test_a_metadata_only_directory_is_not_left_behind(civitdl, tmp_path):
    """It is invisible to GET and DELETE, so nothing else can clean it up."""
    model_dir = tmp_path / "Heime-mid_2805786-vid_3163627"
    civitdl.side_effect = lambda *a, **k: partial_download(tmp_path)

    task_id = utils.create_task(2805786, 3163627)
    with patch.object(utils, "MODEL_ROOT_PATH", str(tmp_path)), \
            patch.dict(utils.MODEL_TYPE_TO_FOLDER, {"lora": str(tmp_path)}), \
            patch.object(utils, "get_safe_metadata", return_value=METADATA), \
            patch("requests.Session.request", return_value=MagicMock(status_code=200)):
        utils._civitdl_async_worker(task_id, 2805786, 3163627, "key")

    assert utils.get_task(task_id)["status"] == "failed"
    assert not model_dir.exists()


def test_a_finished_download_keeps_its_directory(civitdl, tmp_path):
    """The cleanup runs after every download, so it has to know a good one."""
    def wrote_the_model(*args, **kwargs):
        model_dir = partial_download(tmp_path)
        (model_dir / "heime_v10-mid_2805786-vid_3163627.safetensors").write_bytes(b"weights")

    civitdl.side_effect = wrote_the_model

    task_id = utils.create_task(2805786, 3163627)
    with patch.object(utils, "MODEL_ROOT_PATH", str(tmp_path)), \
            patch.dict(utils.MODEL_TYPE_TO_FOLDER, {"lora": str(tmp_path)}), \
            patch.object(utils, "get_safe_metadata", return_value=METADATA):
        utils._civitdl_async_worker(task_id, 2805786, 3163627, "key")

    assert utils.get_task(task_id)["status"] == "finished"
    assert (tmp_path / "Heime-mid_2805786-vid_3163627").exists()


def test_a_downloaded_model_is_never_discarded(tmp_path):
    model_dir = partial_download(tmp_path)
    (model_dir / "heime_v10-mid_2805786-vid_3163627.safetensors").write_bytes(b"weights")

    utils._discard_partial_download(str(model_dir))

    assert model_dir.exists()


def test_an_interrupted_body_does_not_count_as_a_model(tmp_path):
    model_dir = partial_download(tmp_path)
    tmp = model_dir / ".tmp"
    tmp.mkdir()
    (tmp / "heime_v10-mid_2805786-vid_3163627.safetensors").write_bytes(b"half")

    utils._discard_partial_download(str(model_dir))

    assert not model_dir.exists()
