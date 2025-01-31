from app.models import ModelInfo, DownloadResponse
from app.utils import _civitdl, MODEL_ROOT_PATH
from app.utils import find_model_files, civitai_token, delete_model_files

from fastapi import HTTPException, APIRouter
from typing import List


# --- Routers ---
versions_router = APIRouter(
    prefix="/models/{model_id}/versions",
    tags=["versions"]
)

models_router = APIRouter(
    prefix="/models",
    tags=["models"]
)

# --- Endpoints ---
@models_router.get("/", response_model=List[ModelInfo])
def list_models():
    """
    Retrieve a list of all saved models.
    """
    models = find_model_files(None, None)
    return [ModelInfo(**model.__dict__) for model in models]

@models_router.get("/{model_id}", response_model=List[ModelInfo])
def get_model(model_id: int):
    """
    Retrieve model information for the specified model_id.
    """
    models = find_model_files(None, None)
    matched_models = [ModelInfo(**model.__dict__) for model in models if int(model.model_id) == model_id]

    if matched_models:
        return matched_models
    raise HTTPException(status_code=404, detail="Model not found")


@models_router.post("/{model_id}", response_model=DownloadResponse)
def download_model(model_id: int):
    """
    Download the model for the specified model_id.
    """
    result = _civitdl(model_id, version_id=None, api_key=civitai_token)
    return DownloadResponse(**result)

@models_router.delete("/{model_id}", response_model=List[ModelInfo])
def remove_model(model_id: int):
    """
    Delete the model file for the specified model_id and version_id.
    """
    success = delete_model_files(model_id, None)
    if success:
        return [ModelInfo(**model.__dict__) for model in success]
    else:
        raise HTTPException(status_code=404, detail="Model file not found")

@versions_router.get("/{version_id}", response_model=List[ModelInfo])
def get_model_version(model_id: int, version_id: int):
    """
    Retrieve model information for the specified model_id and version_id.
    """
    models = find_model_files(None, None)
    matched_models = [ModelInfo(**model.__dict__) for model in models if int(model.model_id) == model_id and int(model.version_id) == version_id]

    if matched_models:
        return matched_models
    raise HTTPException(status_code=404, detail="Model version not found")

@versions_router.post("/{version_id}", response_model=DownloadResponse)
def download_model_version(model_id: int, version_id: int):
    """
    Download the model for the specified model_id and version_id.
    """
    result = _civitdl(model_id, version_id=version_id, api_key=civitai_token)
    return DownloadResponse(**result)

@versions_router.delete("/{version_id}", response_model=List[ModelInfo])
def remove_model_version(model_id: int, version_id: int):
    """
    Delete the model file for the specified model_id and version_id.
    """
    success = delete_model_files(model_id, version_id)
    if success:
        return [ModelInfo(**model.__dict__) for model in success]
    else:
        raise HTTPException(status_code=404, detail="Model version file not found")

@models_router.delete("/", response_model=List[ModelInfo])
def remove_all_models():
    """
    Delete all saved model files.
    """
    success = delete_model_files(None, None)
    if success:
        return [ModelInfo(**model.__dict__) for model in success]
    else:
        raise HTTPException(status_code=404, detail="No model files found"+MODEL_ROOT_PATH)
