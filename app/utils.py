import json
import os
import re
import sys
import shutil
import requests
import uuid
import threading

from contextlib import contextmanager
from typing import Callable, List, Dict, Any, Optional
from fastapi import HTTPException

from app.models import ModelInfo
from app.sorter import sort_model
from helpers.core.utils import APIException
from helpers.sourcemanager import SourceManager

from civitdl.args.argparser import get_args
from civitdl.batch._metadata import Metadata
from civitdl.batch.batch_download import batch_download, BatchOptions

MODEL_ROOT_PATH = os.getenv("MODEL_ROOT_PATH", "/data")
CIVITAI_TOKEN = os.getenv("CIVITAI_TOKEN", "")

# civitdl never passes a timeout, so a Civitai connection that goes quiet used
# to block the download thread for the life of the process. The read timeout is
# how long a transfer may go without receiving *anything*, not how long it may
# take: a 12GB file over a slow link is fine as long as bytes keep arriving.
# Five minutes of silence is a dead connection, not a slow one -- and civitdl
# restarts an aborted download from zero, so erring short is expensive.
CIVITAI_TIMEOUT = (
    float(os.getenv("CIVITAI_CONNECT_TIMEOUT", "15")),
    float(os.getenv("CIVITAI_READ_TIMEOUT", "300")),
)

# How long a download may wait for another download of the same version. This
# is a backstop against a hold nothing else can explain, not a stall detector:
# the read timeout is what ends a stalled download, in minutes. It has to stay
# clear of a legitimately long download -- 12GB over a slow link takes hours,
# and failing the second request for it with "it is stuck" would be a lie.
DOWNLOAD_LOCK_TIMEOUT = float(os.getenv("DOWNLOAD_LOCK_TIMEOUT", "86400"))

MODEL_FILE_PATTERN = re.compile(
    r".*-mid_(\d+)(?:-vid_(\d+))?.*\.(safetensors|ckpt|pt)$"
)

MODEL_TYPE_TO_FOLDER: Dict[str, str] = {
    "lora": os.path.join(MODEL_ROOT_PATH, "models", "Lora"),
    "locon": os.path.join(MODEL_ROOT_PATH, "models", "Lora"),
    "dora": os.path.join(MODEL_ROOT_PATH, "models", "Lora"),
    "vae": os.path.join(MODEL_ROOT_PATH, "models", "VAE"),
    "checkpoint": os.path.join(MODEL_ROOT_PATH, "models", "Stable-diffusion"),
    "textualinversion": os.path.join(MODEL_ROOT_PATH, "embeddings"),
}

DOWNLOAD_REFUSED = "Unable to download this model as it requires a valid API Key."
RATE_LIMITED = "Civitai is rate limiting this client. Try again later."


class _CivitaiSession(requests.Session):
    """
    Session that remembers why Civitai refused a download, and that gives up.

    civitdl reports every refusal as a missing API key, discarding the reason
    Civitai gave, so keep it from the response civitdl already made. It also
    never passes a timeout, so add one: a stalled connection used to keep the
    download thread alive with nothing to show for it until the process died.
    """

    refusal = None
    transport_error = None

    def request(self, method, url, **kwargs):
        # civitdl calls session.get() without a timeout, both for the metadata
        # and for the streamed model body, so supply one here.
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = CIVITAI_TIMEOUT
        try:
            response = super().request(method, url, **kwargs)
        except requests.RequestException as error:
            self._remember(error)
            raise
        return self._watch_body(response)

    def _remember(self, error: Exception) -> None:
        # civitdl's retry loop swallows these and returns as if it had done its
        # job. Without keeping one, a download that timed out is reported as a
        # missing API key.
        self.transport_error = f"{type(error).__name__} from Civitai: {error}"

    def _watch_body(self, response):
        """Keep the reason a streamed body stopped arriving.

        A download stalls mid-file, not while connecting, and that raises out
        of iter_content -- past the try above, where nothing would see it.
        """
        streamed = response.iter_content

        def iter_content(*args, **kwargs):
            try:
                for chunk in streamed(*args, **kwargs):
                    yield chunk
            except requests.RequestException as error:
                self._remember(error)
                raise

        response.iter_content = iter_content
        return response

    def get(self, url, **kwargs):
        response = super().get(url, **kwargs)
        if "/api/download/models/" in str(url) and response.status_code in (401, 403):
            try:
                self.refusal = response.json().get("message")
            except ValueError:
                pass
        return response


# Task management for async downloads
_download_tasks: Dict[str, Dict[str, Any]] = {}
_tasks_lock = threading.Lock()

# One lock per model version. Two requests for the same version would both see
# no local file and both start civitdl, which writes to the same .tmp path.
_download_locks: Dict[tuple, threading.Lock] = {}
_download_locks_guard = threading.Lock()


def _download_lock(model_id: int, version_id: int) -> threading.Lock:
    """Get the lock that serializes downloads of one model version."""
    with _download_locks_guard:
        return _download_locks.setdefault((model_id, version_id), threading.Lock())


@contextmanager
def _hold_download_lock(model_id: int, version_id: int):
    """Serialize downloads of one version, but never wait on it forever.

    A download that hung held this lock for the life of the process, and every
    later request for the same version blocked here *before* creating anything:
    no directory, no `.tmp`, no civitdl log line, no error -- the task just sat
    at "downloading" until the container was restarted.
    """
    lock = _download_lock(model_id, version_id)
    if not lock.acquire(timeout=DOWNLOAD_LOCK_TIMEOUT):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Another download of model {model_id} version {version_id} has "
                f"held the download lock for over {int(DOWNLOAD_LOCK_TIMEOUT)}s. "
                "It is stuck; restart the service to clear it."
            ),
        )
    try:
        yield
    finally:
        lock.release()


def _discard_partial_download(model_dir: str) -> None:
    """Delete a model directory that never got a model file.

    civitdl writes the metadata and the preview images before the model itself,
    so a download that dies in between leaves a directory the rest of the API
    cannot see: `find_model_files` only matches model files, so GET and DELETE
    both 404 while the directory sits on disk and confuses the next attempt.
    """
    if not model_dir or not os.path.isdir(model_dir):
        return

    for root, _, files in os.walk(model_dir):
        if ".tmp" in root:
            continue  # an interrupted body, not a model file
        if any(MODEL_FILE_PATTERN.match(file) for file in files):
            return

    print(f"Discarding partial download at {model_dir}.")
    shutil.rmtree(model_dir, ignore_errors=True)


def create_task_id() -> str:
    """Generate a unique task ID."""
    return str(uuid.uuid4())


def create_task(model_id: int, version_id: Optional[int] = None) -> str:
    """Create a new download task and return its ID."""
    task_id = create_task_id()
    with _tasks_lock:
        _download_tasks[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0,
            "model_id": model_id,
            "version_id": version_id,
            "model_name": None,
            "model_type": None,
            "result": None,
            "error": None
        }
    return task_id


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status by task ID."""
    with _tasks_lock:
        return _download_tasks.get(task_id)


def update_task(task_id: str, **kwargs) -> None:
    """Update task status."""
    with _tasks_lock:
        if task_id in _download_tasks:
            _download_tasks[task_id].update(kwargs)


def _model_dir(metadata: Dict[str, Any], output_dir: str) -> str:
    """Where civitdl's sorter will put this model version."""
    return sort_model(
        metadata.get("model_dict") or {},
        metadata.get("version_dict") or {},
        "",
        output_dir
    ).model_dir_path


def _get_tmp_file_size(base_dir: str) -> int:
    """Get total size of files in .tmp directories under base_dir."""
    total_size = 0
    for root, dirs, files in os.walk(base_dir):
        if ".tmp" in root:
            for file in files:
                try:
                    total_size += os.path.getsize(os.path.join(root, file))
                except OSError:
                    pass
    return total_size


def get_available_disk_space(path: str) -> int:
    """Get available disk space in bytes for the given path."""
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def _primary_file_size(metadata: Dict[str, Any]) -> int:
    """Size in bytes of the file civitdl will download for this version.

    The requested version is not always the first entry of modelVersions, so
    reading the files off modelVersions[0] sized whichever version Civitai
    happened to list first.
    """
    files = (metadata.get("version_dict") or {}).get("files") or []

    if not files:
        version_id = str(metadata.get("version_id", ""))
        model_dict = metadata.get("model_dict") or {}
        for version in model_dict.get("modelVersions", []):
            if str(version.get("id")) == version_id:
                files = version.get("files") or []
                break

    for file in files:
        if file.get("primary", False):
            return int(file.get("sizeKB", 0) * 1024)

    return int(files[0].get("sizeKB", 0) * 1024) if files else 0


def get_model_file_size(model_id: int, version_id: Optional[int] = None) -> int:
    """Get expected file size in bytes from Civitai API metadata."""
    if version_id:
        model_id_str = f"civitai.com/models/{model_id}?modelVersionId={version_id}"
    else:
        model_id_str = str(model_id)

    return _primary_file_size(get_safe_metadata(model_id_str))


def check_disk_space(model_id: int, version_id: Optional[int] = None) -> None:
    """
    Check if there is enough disk space to download the model.
    Raises HTTPException with 507 status code if insufficient space.
    """
    try:
        # Already downloaded models don't require additional space, skip the check
        if find_model_files(model_id, version_id):
            return

        file_size = get_model_file_size(model_id, version_id)
        if file_size == 0:
            return  # Cannot determine size, proceed with download

        available_space = get_available_disk_space(MODEL_ROOT_PATH)

        # Add 10% margin for safety
        required_space = int(file_size * 1.1)

        if available_space < required_space:
            available_mb = available_space / (1024 * 1024)
            required_mb = required_space / (1024 * 1024)
            raise HTTPException(
                status_code=507,
                detail=f"Insufficient storage. Required: {required_mb:.1f}MB, Available: {available_mb:.1f}MB"
            )
    except HTTPException:
        raise
    except Exception:
        pass  # If check fails, proceed with download


def wrap_cli_args(
    cli_func: Callable[[], Dict[str, Any]],
    required_args: List[str],
    **override_kwargs
) -> Dict[str, Any]:
    """
    Temporarily replace CLI arguments, execute `cli_func`, and return the resulting dictionary with `override_kwargs` applied.

    **Description:**
    This function allows you to simulate CLI arguments by temporarily modifying `sys.argv`, executing the provided CLI function, and then restoring the original `sys.argv`. Additionally, it overrides specific keyword arguments in the result.

    **Parameters:**
    - `cli_func` (`Callable[[], Dict[str, Any]]`): Function for CLI that takes arguments.
    - `required_args` (`List[str]`): List of arguments to set in `sys.argv`.
    - `override_kwargs` (`Dict[str, Any]`): Keyword arguments to override the execution result.

    **Returns:**
    - `Dict[str, Any]`: Dictionary after executing `cli_func` with overridden keyword arguments.

    **Example:**
    ```python
    result = wrap_cli_args(cli_function, ['--model', '123'], verbose=True)
    ```
    """
    original_argv = sys.argv
    try:
        sys.argv = ["cli_tool"] + required_args
        result_dict = cli_func()

        for key, value in override_kwargs.items():
            if key in result_dict:
                result_dict[key] = value

    finally:
        sys.argv = original_argv

    return result_dict


def get_safe_metadata(model_str: str) -> Dict[str, Any]:
    """
    Retrieve metadata for the model specified by `model_str` and return it in a safe format with non-built-in types converted to strings.

    **Description:**
    This function parses the model string to extract the model ID, retrieves metadata from the API, and ensures that all data types in the metadata are JSON serializable.

    **Parameters:**
    - `model_str` (`str`): Model specification string in the format `"civitai.com/models/xxx"`.

    **Returns:**
    - `Dict[str, Any]`: Dictionary containing the serialized metadata.

    **Raises:**
    - `AssertionError`: If the retrieved `model_id` does not match the parsed ID.

    **Example:**
    ```python
    metadata = get_safe_metadata("civitai.com/models/12345")
    ```
    """
    source_manager = SourceManager()
    parsed_id = source_manager.parse_src([model_str])[0]

    metadata = Metadata(
        nsfw_mode="0",
        max_images=0,
        session=_CivitaiSession()
    ).make_api_call(parsed_id)

    assert metadata.model_id == parsed_id.model_id, f"Model {parsed_id} not found."

    def _serialize(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)

    return json.loads(json.dumps(metadata.__dict__, default=_serialize))


def _read_extra_data(extra_data_path: str, version_id: Optional[int]) -> dict:
    """Read the metadata civitdl saved next to a model file.

    Everything a `ModelInfo` knows beyond its path comes from this one JSON, so
    nothing here talks to Civitai. A directory with no `extra_data` still has to
    list -- that is what `model_type="unknown"` is for -- so every field falls
    back to a value the response model accepts.
    """
    extra = {
        "model_type": "unknown",
        "name": "",
        "description": "",
        "created_at": "",
    }
    if not os.path.exists(extra_data_path):
        return extra

    with open(extra_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    extra["model_type"] = data.get("type", "").lower()
    extra["name"] = data.get("name", "")
    extra["description"] = data.get("description", "")
    extra["tags"] = data.get("tags") or []
    extra["nsfw"] = data.get("nsfw")
    extra["nsfw_level"] = data.get("nsfwLevel")
    extra["creator"] = (data.get("creator") or {}).get("username")

    # Find the version in modelVersions array
    version = next(
        (v for v in data.get("modelVersions", [])
         if v.get("id") == version_id),
        None,
    )
    if version is None:
        return extra

    extra["created_at"] = version.get("createdAt", "")
    extra["published_at"] = version.get("publishedAt")
    extra["base_model"] = version.get("baseModel")
    extra["base_model_type"] = version.get("baseModelType")
    extra["version_name"] = version.get("name")
    extra["version_description"] = version.get("description")
    extra["trained_words"] = version.get("trainedWords") or []

    stats = version.get("stats") or {}
    extra["download_count"] = stats.get("downloadCount")
    extra["thumbs_up_count"] = stats.get("thumbsUpCount")

    # civitdl only ever downloads the primary file, so that is the one whose
    # size and hash describe what is actually on disk.
    files = version.get("files") or []
    primary = next((f for f in files if f.get("primary")), None)
    if primary is None and files:
        primary = files[0]
    if primary:
        extra["file_size_kb"] = primary.get("sizeKB")
        extra["sha256"] = (primary.get("hashes") or {}).get("SHA256")

    return extra


def find_model_files(
    model_id: Optional[int] = None,
    version_id: Optional[int] = None
) -> List[ModelInfo]:
    """
    Recursively search for files matching the specified `model_id` and `version_id`, and return a list of `ModelInfo`.
    If both are `None`, search all models.

    **Description:**
    This function traverses the `MODEL_ROOT_PATH` directory, matches files based on the naming pattern, and collects metadata about each model file found.

    **Parameters:**
    - `model_id` (`Optional[int]`): Model ID (`None` to target all models).
    - `version_id` (`Optional[int]`): Version ID (`None` to target all versions).

    **Returns:**
    - `List[ModelInfo]`: List of `ModelInfo` objects matching the criteria.

    **Example:**
    ```python
    all_models = find_model_files()
    specific_model = find_model_files(model_id=12345)
    specific_version = find_model_files(model_id=12345, version_id=1)
    ```
    """
    found_models = []

    for root, _, files in os.walk(MODEL_ROOT_PATH):
        if ".tmp" in root:
            continue

        for file in files:
            match = MODEL_FILE_PATTERN.match(file)
            if not match:
                continue

            found_model_id = int(match.group(1))
            found_version_id = int(match.group(2)) if match.group(2) else None

            if (model_id is None or model_id == found_model_id) and \
               (version_id is None or version_id == found_version_id):
                extra_data_path = os.path.join(
                    root,
                    f"extra_data-vid_{found_version_id}",
                    f"model_dict-mid_{found_model_id}-vid_{found_version_id}.json"
                )

                extra = _read_extra_data(extra_data_path, found_version_id)

                found_models.append(
                    ModelInfo(
                        model_id=found_model_id,
                        version_id=found_version_id,
                        model_dir=root,
                        filename=file,
                        **extra
                    )
                )

    return found_models


def delete_model_files(
    model_id: Optional[int] = None,
    version_id: Optional[int] = None
) -> List[ModelInfo]:
    """
    Recursively delete model files and directories matching the specified `model_id` and `version_id`.

    **Description:**
    This function identifies model files based on the provided IDs and removes their corresponding directories from the filesystem.

    **Parameters:**
    - `model_id` (`Optional[int]`): Model ID.
    - `version_id` (`Optional[int]`): Version ID.

    **Returns:**
    - `List[ModelInfo]`: List of `ModelInfo` objects that were targeted for deletion.

    **Example:**
    ```python
    deleted_models = delete_model_files(model_id=12345, version_id=1)
    ```
    """
    models_to_delete = find_model_files(model_id, version_id)
    if not models_to_delete:
        return []

    for model_info in models_to_delete:
        shutil.rmtree(model_info.model_dir, ignore_errors=True)

    return models_to_delete


def _civitdl(
    model_id: int,
    version_id: Optional[int] = None,
    api_key: Optional[str] = None
) -> ModelInfo:
    """
    Download a model from Civitai by specifying `model_id` and `version_id`.
    Returns HTTP 200 if a model with the same `model_id` and `version_id` already exists.

    **Description:**
    This function handles the downloading of a specific model version from Civitai. It first checks if the model version already exists to prevent duplicate downloads. If not, it retrieves the model metadata, prepares the download arguments, and initiates the batch download process.

    **Parameters:**
    - `model_id` (`int`): Model ID.
    - `version_id` (`Optional[int]`): Version ID.
    - `api_key` (`Optional[str]`): Civitai API Key (if not provided, environment variables will be used).

    **Returns:**
    - `ModelInfo`: Information about the downloaded model.

    **Raises:**
    - `HTTPException`:
        - `404`: If the model is not found or if the download verification fails.

    **Example:**
    ```python
    downloaded_model = _civitdl(model_id=12345, version_id=1, api_key="your_api_key")
    ```
    """
    existing_models = find_model_files(model_id, version_id)
    if len(existing_models) >= 1:
        return existing_models[0]

    if version_id:
        model_id_str = f"civitai.com/models/{model_id}?modelVersionId={version_id}"
    else:
        model_id_str = str(model_id)

    try:
        metadata = get_safe_metadata(model_id_str)
        model_type = metadata.get("model_dict", {}).get("type", "").lower()
        output_dir = MODEL_TYPE_TO_FOLDER.get(model_type)

        args = wrap_cli_args(
            get_args,
            [model_id_str, output_dir or MODEL_ROOT_PATH],
            api_key=api_key,
            retry_count=1,
            pause_time=0.0,
            with_color=False,
            verbose=False,
            sorter=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "sorter.py"
            )
        )
        print(f"Downloading model {model_id_str} with args: { {k: '****' if k == 'api_key' else v for k, v in args.items()} }")
        source_strings = args.pop("source_strings", None)
        root_dir = args.pop("rootdir", None)

        resolved_model_id = int(metadata["model_id"])
        resolved_version_id = int(metadata["version_id"])
        session = _CivitaiSession()

        with _hold_download_lock(resolved_model_id, resolved_version_id):
            # A concurrent request for the same version may have finished it
            # while this one waited for the lock.
            if not find_model_files(resolved_model_id, resolved_version_id):
                batch_options = BatchOptions(**args)
                batch_options.session = session

                batch_download(
                    source_strings=source_strings,
                    rootdir=root_dir if root_dir != "None" else None,
                    batchOptions=batch_options
                )
                print(f"Model {model_id_str} has been successfully downloaded to {output_dir}.")

            downloaded = find_model_files(resolved_model_id, resolved_version_id)
            _discard_partial_download(
                _model_dir(metadata, output_dir or MODEL_ROOT_PATH))

        if len(downloaded) == 0:
            raise HTTPException(
                status_code=401,
                detail=session.refusal or session.transport_error or DOWNLOAD_REFUSED
            )
        if len(downloaded) > 1:
            raise HTTPException(status_code=500, detail="Unexpected error occurred.")
        return downloaded[0]

    except APIException as e:
        if e.status_code == 429:
            raise HTTPException(status_code=429, detail=RATE_LIMITED) from e
        raise HTTPException(status_code=404, detail="Model not found on Civitai.") from e
    except AssertionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _civitdl_async_worker(
    task_id: str,
    model_id: int,
    version_id: Optional[int] = None,
    api_key: Optional[str] = None
) -> None:
    """
    Worker function to download a model asynchronously and update task status.
    This function runs in a background thread with real-time progress tracking
    by monitoring file size in .tmp directories.
    """
    download_complete = threading.Event()
    download_error = [None]  # Use list to allow modification in nested function
    session = _CivitaiSession()

    def do_download():
        """Execute batch_download in a separate thread."""
        try:
            if version_id:
                model_id_str = f"civitai.com/models/{model_id}?modelVersionId={version_id}"
            else:
                model_id_str = str(model_id)

            metadata = get_safe_metadata(model_id_str)
            model_type = metadata.get("model_dict", {}).get("type", "").lower()
            output_dir = MODEL_TYPE_TO_FOLDER.get(model_type)

            args = wrap_cli_args(
                get_args,
                [model_id_str, output_dir or MODEL_ROOT_PATH],
                api_key=api_key,
                retry_count=1,
                pause_time=0.0,
                with_color=False,
                verbose=False,
                sorter=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "sorter.py"
                )
            )
            source_strings = args.pop("source_strings", None)
            root_dir = args.pop("rootdir", None)

            with _hold_download_lock(resolved_model_id, resolved_version_id):
                # A concurrent request for the same version may have finished
                # it while this one waited for the lock.
                if find_model_files(resolved_model_id, resolved_version_id):
                    return

                batch_options = BatchOptions(**args)
                batch_options.session = session

                try:
                    batch_download(
                        source_strings=source_strings,
                        rootdir=root_dir if root_dir != "None" else None,
                        batchOptions=batch_options
                    )
                finally:
                    # No find_model_files() guard here: it walks the whole
                    # library, and _discard_partial_download already keeps a
                    # directory that has a model file in it.
                    _discard_partial_download(
                        _model_dir(metadata, output_dir or MODEL_ROOT_PATH))
        except Exception as e:
            download_error[0] = e
        finally:
            download_complete.set()

    try:
        # Check if model already exists
        update_task(task_id, status="downloading", progress=1)
        existing_models = find_model_files(model_id, version_id)
        if len(existing_models) >= 1:
            existing = existing_models[0]
            update_task(
                task_id,
                status="finished",
                progress=100,
                version_id=existing.version_id,
                model_name=existing.name or None,
                model_type=existing.model_type.value,
                result=existing
            )
            return

        # Get metadata for file size estimation
        if version_id:
            model_id_str = f"civitai.com/models/{model_id}?modelVersionId={version_id}"
        else:
            model_id_str = str(model_id)

        metadata = get_safe_metadata(model_id_str)
        model_dict = metadata.get("model_dict", {})
        model_type = model_dict.get("type", "").lower()
        output_dir = MODEL_TYPE_TO_FOLDER.get(model_type, MODEL_ROOT_PATH)
        resolved_model_id = int(metadata["model_id"])
        resolved_version_id = int(metadata["version_id"])
        # A request that omitted the version now knows which one it resolved to.
        update_task(
            task_id,
            version_id=resolved_version_id,
            model_name=model_dict.get("name") or None,
            model_type=model_type or None
        )

        # Get expected file size from metadata, for the version asked for
        expected_size = _primary_file_size(metadata)

        update_task(task_id, progress=5)

        # Start download in separate thread
        download_thread = threading.Thread(target=do_download)
        download_thread.start()

        # Monitor progress by checking .tmp file sizes
        while not download_complete.is_set():
            if expected_size > 0:
                current_size = _get_tmp_file_size(_model_dir(metadata, output_dir))
                progress = min(5 + int((current_size / expected_size) * 90), 95)
                update_task(task_id, progress=progress)
            download_complete.wait(timeout=0.5)

        # Wait for download thread to finish
        download_thread.join()

        # Check for errors
        if download_error[0]:
            raise download_error[0]

        update_task(task_id, progress=98)

        # Verify download
        downloaded = find_model_files(
            int(metadata["model_id"]),
            int(metadata["version_id"])
        )

        if len(downloaded) == 0:
            update_task(
                task_id,
                status="failed",
                progress=0,
                error=session.refusal or session.transport_error or DOWNLOAD_REFUSED
            )
            return

        # Success
        update_task(
            task_id,
            status="finished",
            progress=100,
            result=downloaded[0]
        )

    except APIException as e:
        update_task(
            task_id,
            status="failed",
            progress=0,
            error=RATE_LIMITED if e.status_code == 429 else "Model not found on Civitai."
        )
    except AssertionError as e:
        update_task(
            task_id,
            status="failed",
            progress=0,
            error=str(e)
        )
    except HTTPException as e:
        update_task(
            task_id,
            status="failed",
            progress=0,
            error=e.detail
        )
    except Exception as e:
        print(f"Download failed for model {model_id}: {e}")
        update_task(
            task_id,
            status="failed",
            progress=0,
            error=str(e)
        )
