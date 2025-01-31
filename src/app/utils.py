
from app.models import ModelInfo

from civitconfig.data.configmanager import ConfigManager
from civitdl.args.argparser import get_args
from civitdl.batch._metadata import Metadata
from civitdl.batch.batch_download import batch_download, BatchOptions
from fastapi import HTTPException
from helpers.core.utils import APIException
from helpers.sourcemanager import SourceManager
from types import SimpleNamespace
from typing import Callable, List, Dict, Any
from typing import List, Optional
import json
import os
import re
import requests
import shutil
import sys

# --- 環境変数 & ディレクトリマッピング ---
MODEL_ROOT_PATH = os.getenv("MODEL_ROOT_PATH", "/data")

MODEL_TYPE_TO_FOLDER: Dict[str, str] = {
    "lora": os.path.join(MODEL_ROOT_PATH, "Lora"),
    "locon": os.path.join(MODEL_ROOT_PATH, "Lora"),
    "vae": os.path.join(MODEL_ROOT_PATH, "VAE"),
    "checkpoint": os.path.join(MODEL_ROOT_PATH, "Stable-diffusion"),
    "textualinversion": os.path.join(MODEL_ROOT_PATH, "text_encoder"),
}


c = ConfigManager()
c._setFallback()
civitai_token=os.getenv("CIVITAI_TOKEN","")


def wrap_cli_args(cli_func: Callable[[], Dict[str, Any]], required_args: List[str], **kwargs):
    original_argv = sys.argv
    try:
        sys.argv = ["cli_tool"] + required_args
        result_dict = cli_func()
        for key, value in kwargs.items():
            if key in result_dict:
                result_dict[key] = value
    finally:
        sys.argv = original_argv
    return SimpleNamespace(**result_dict)


def get_safe_metadata(_id: str):
    """ Retrieve Metadata for the specified id and return a safe dictionary """
    source_manager = SourceManager()
    _id = (source_manager.parse_src([_id]))[0]
    metadata = Metadata(
        nsfw_mode="0",
        max_images=0,
        session=requests.session()
    ).make_api_call(_id)
    assert metadata.model_id == _id.model_id, f"Model {_id} not found."
    def serialize(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)

    return json.loads(json.dumps(metadata.__dict__, default=serialize))


def find_model_files(model_id: Optional[int] = None, version_id: Optional[int] = None) -> List[ModelInfo]:
    """Find model files matching model_id and version_id. If both are None, return all files."""
    model_pattern = re.compile(r".*-mid_(\d+)(?:-vid_(\d+))?.*\.(safetensors|ckpt|pt)$")
    models = []

    for root, _, files in os.walk(MODEL_ROOT_PATH):
        for file in files:
            match = model_pattern.match(file)
            if match:
                found_model_id = int(match.group(1))
                found_version_id = int(match.group(2)) if match.group(2) else None

                # model_id, version_id でフィルタリング
                if (model_id is None or model_id == found_model_id) and (version_id is None or version_id == found_version_id):
                    extra_data_dir_path = os.path.join(
                        root,
                        f"extra_data-vid_{found_version_id}",
                        f"model_dict-mid_{found_model_id}-vid_{found_version_id}.json"
                    )

                    model_type = "unknown"
                    if os.path.exists(extra_data_dir_path):
                        with open(extra_data_dir_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            model_type = data.get("type", "").lower()

                    models.append(ModelInfo(
                        model_id=found_model_id,
                        version_id=found_version_id,
                        model_dir=root,
                        filename=file,
                        model_type=model_type,
                    ))

    return models


def delete_model_files(model_id: Optional[int] = None, version_id: Optional[int] = None) -> List[ModelInfo]:
    """Delete model files and corresponding directories recursively"""
    models_to_delete = find_model_files(model_id, version_id)
    # print(models_to_delete)
    if not models_to_delete:
        print("@@@@@@@@@")
        return []  # No matching models found

    for model in models_to_delete:
        print("deleteing...", model)
        shutil.rmtree(model.model_dir, ignore_errors=True)  # Recursively delete model directory

    return models_to_delete


def _civitdl(model_id, version_id=None, api_key=None):
    paths = find_model_files(None, None)
    for model in paths:
        if model.model_id == model_id and model.version_id == version_id:
            raise HTTPException(status_code=304, detail="Model already downloaded.")

    model_id_str = f"civitai.com/models/{model_id}?modelVersionId={version_id}" if version_id else str(model_id)
    try:
        metadata = get_safe_metadata(model_id_str)
        model_type = metadata.get("model_dict", {}).get("type", "").lower()
        output_dir = MODEL_TYPE_TO_FOLDER.get(model_type)
        args = wrap_cli_args(get_args,
                            [model_id_str, output_dir],
                            api_key=api_key,
                            retry_count=1,
                            pause_time=0.,
                            with_color=False,
                            verbose=False,
                            sorter=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sorter.py")
                            )
        source_strings, root_dir = args.__dict__.pop("source_strings", None), args.__dict__.pop("rootdir", None)
        batch_download(
            source_strings=source_strings,
            rootdir=root_dir,
            batchOptions=BatchOptions(**args.__dict__)
        )
        return {"model_id": int(metadata["model_id"]), "version_id": int(metadata["version_id"]), "model_dir": output_dir, "model_type": model_type}
    except APIException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AssertionError as e:
        raise HTTPException(status_code=404, detail=str(e))
