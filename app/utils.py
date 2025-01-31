import json
import os
import re
import sys
import shutil
import requests

from typing import Callable, List, Dict, Any, Optional
from fastapi import HTTPException

from app.models import ModelInfo
from helpers.core.utils import APIException
from helpers.sourcemanager import SourceManager

from civitconfig.data.configmanager import ConfigManager
from civitdl.args.argparser import get_args
from civitdl.batch._metadata import Metadata
from civitdl.batch.batch_download import batch_download, BatchOptions

MODEL_ROOT_PATH = os.getenv("MODEL_ROOT_PATH", "/data")
CIVITAI_TOKEN = os.getenv("CIVITAI_TOKEN", "")

MODEL_TYPE_TO_FOLDER: Dict[str, str] = {
    "lora": os.path.join(MODEL_ROOT_PATH, "Lora"),
    "locon": os.path.join(MODEL_ROOT_PATH, "Lora"),
    "vae": os.path.join(MODEL_ROOT_PATH, "VAE"),
    "checkpoint": os.path.join(MODEL_ROOT_PATH, "Stable-diffusion"),
    "textualinversion": os.path.join(MODEL_ROOT_PATH, "text_encoder"),
}

config_manager = ConfigManager()
config_manager._setFallback()


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
        session=requests.session()
    ).make_api_call(parsed_id)

    assert metadata.model_id == parsed_id.model_id, f"Model {parsed_id} not found."

    def _serialize(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)

    return json.loads(json.dumps(metadata.__dict__, default=_serialize))


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
    model_pattern = re.compile(
        r".*-mid_(\d+)(?:-vid_(\d+))?.*\.(safetensors|ckpt|pt)$"
    )
    found_models = []

    for root, _, files in os.walk(MODEL_ROOT_PATH):
        if ".tmp" in root:
            continue

        for file in files:
            match = model_pattern.match(file)
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

                model_type = "unknown"
                if os.path.exists(extra_data_path):
                    with open(extra_data_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        model_type = data.get("type", "").lower()

                found_models.append(
                    ModelInfo(
                        model_id=found_model_id,
                        version_id=found_version_id,
                        model_dir=root,
                        filename=file,
                        model_type=model_type
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
    Returns HTTP 304 if a model with the same `model_id` and `version_id` already exists.

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
        - `304`: If the model is already downloaded.
        - `404`: If the model is not found or if the download verification fails.

    **Example:**
    ```python
    downloaded_model = _civitdl(model_id=12345, version_id=1, api_key="your_api_key")
    ```
    """
    existing_files = find_model_files(model_id, None)
    if any(m.version_id == version_id for m in existing_files):
        raise HTTPException(status_code=304, detail="Model already downloaded.")

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
            [model_id_str, output_dir],
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

        batch_download(
            source_strings=source_strings,
            rootdir=root_dir,
            batchOptions=BatchOptions(**args)
        )

        downloaded = find_model_files(
            int(metadata["model_id"]),
            int(metadata["version_id"])
        )

        assert len(downloaded) == 1, "Downloaded model could not be verified."
        return downloaded[0]

    except APIException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AssertionError as e:
        raise HTTPException(status_code=404, detail=str(e))
