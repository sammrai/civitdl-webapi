from typing import List

from fastapi import APIRouter, HTTPException

from app.models import ModelInfo
from app.utils import (
    _civitdl,
    CIVITAI_TOKEN,
    delete_model_files,
    find_model_files,
    MODEL_ROOT_PATH,
)

# --- Routers ---
versions_router = APIRouter(
    prefix="/models/{model_id}/versions",
    tags=["versions"],
)

models_router = APIRouter(
    prefix="/models",
    tags=["models"],
)

# --- Endpoints ---

@models_router.get("/", response_model=List[ModelInfo])
def list_all_models():
    """
    Retrieve a list of all saved models.

    This endpoint fetches all model files available in the system and returns their information.

    **Returns:**
    - `List[ModelInfo]`: A list of `ModelInfo` objects representing each saved model.

    **Raises:**
    - `HTTPException`: If an unexpected error occurs while fetching models.
    """
    models = find_model_files(model_id=None, version_id=None)
    return [ModelInfo(**model.__dict__) for model in models]


@models_router.get("/{model_id}", response_model=List[ModelInfo])
def get_model(model_id: int):
    """
    Retrieve information for the specified model ID.

    This endpoint fetches all versions of a model by its ID.

    **Parameters:**
    - `model_id` (int): The unique identifier of the model to retrieve.

    **Returns:**
    - `List[ModelInfo]`: A list of `ModelInfo` objects matching the specified model ID.

    **Raises:**
    - `HTTPException`: If no model is found with the given ID.
    """
    models = find_model_files(model_id=model_id, version_id=None)
    matched_models = [
        ModelInfo(**model.__dict__) for model in models if int(model.model_id) == model_id
    ]

    if matched_models:
        return matched_models
    raise HTTPException(status_code=404, detail="Model not found")


@models_router.post("/{model_id}", response_model=ModelInfo)
def download_model(model_id: int):
    """
    Download the latest version of the specified model ID.

    This endpoint initiates the download process for the latest version of the given model.

    **Parameters:**
    - `model_id` (int): The unique identifier of the model to download.

    **Returns:**
    - `ModelInfo`: Information about the downloaded model.

    **Raises:**
    - `HTTPException`: If the download process fails.
    """
    return _civitdl(model_id=model_id, version_id=None, api_key=CIVITAI_TOKEN)


@models_router.delete("/{model_id}", response_model=List[ModelInfo])
def remove_model(model_id: int):
    """
    Delete all versions of the specified model ID.

    This endpoint removes all files associated with the given model ID from the system.

    **Parameters:**
    - `model_id` (int): The unique identifier of the model to delete.

    **Returns:**
    - `List[ModelInfo]`: A list of `ModelInfo` objects representing the deleted models.

    **Raises:**
    - `HTTPException`: If no model files are found for the given ID.
    """
    deleted_models = delete_model_files(model_id=model_id, version_id=None)
    if deleted_models:
        return [ModelInfo(**model.__dict__) for model in deleted_models]
    else:
        raise HTTPException(status_code=404, detail="Model file not found")


@models_router.delete("/", response_model=List[ModelInfo])
def remove_all_models():
    """
    Delete all saved model files.

    This endpoint removes all model files from the system.

    **Returns:**
    - `List[ModelInfo]`: A list of `ModelInfo` objects representing the deleted models.

    **Raises:**
    - `HTTPException`: If no model files are found.
    """
    deleted_models = delete_model_files(model_id=None, version_id=None)
    if deleted_models:
        return [ModelInfo(**model.__dict__) for model in deleted_models]
    else:
        detail_message = f"No model files found in {MODEL_ROOT_PATH}"
        raise HTTPException(status_code=404, detail=detail_message)


@versions_router.get("/{version_id}", response_model=ModelInfo)
def get_model_version(model_id: int, version_id: int):
    """
    Retrieve information for the specified model ID and version ID.

    This endpoint fetches details of a specific version of a model.

    **Parameters:**
    - `model_id` (int): The unique identifier of the model.
    - `version_id` (int): The unique identifier of the model version.

    **Returns:**
    - `ModelInfo`: Information about the specified model version.

    **Raises:**
    - `HTTPException`: 
        - `404`: If the model version is not found.
        - `500`: If multiple entries are found, indicating a server error.
    """
    models = find_model_files(model_id=model_id, version_id=version_id)
    if len(models) == 1:
        return models[0]
    elif len(models) == 0:
        raise HTTPException(status_code=404, detail="Model version file not found")
    else:
        raise HTTPException(status_code=500, detail="Multiple model version files found")


@versions_router.post("/{version_id}", response_model=ModelInfo)
def download_model_version(model_id: int, version_id: int):
    """
    Download the specified version of the model.

    This endpoint initiates the download process for a specific version of a model.

    **Parameters:**
    - `model_id` (int): The unique identifier of the model.
    - `version_id` (int): The unique identifier of the model version to download.

    **Returns:**
    - `ModelInfo`: Information about the downloaded model version.

    **Raises:**
    - `HTTPException`: If the download process fails.
    """
    return _civitdl(model_id=model_id, version_id=version_id, api_key=CIVITAI_TOKEN)


@versions_router.delete("/{version_id}", response_model=ModelInfo)
def remove_model_version(model_id: int, version_id: int):
    """
    Delete the specified version of the model.

    This endpoint removes a specific version of a model from the system.

    **Parameters:**
    - `model_id` (int): The unique identifier of the model.
    - `version_id` (int): The unique identifier of the model version to delete.

    **Returns:**
    - `ModelInfo`: Information about the deleted model version.

    **Raises:**
    - `HTTPException`: 
        - `404`: If the model version file is not found.
        - `500`: If multiple entries are found, indicating a server error.
    """
    deleted_models = delete_model_files(model_id=model_id, version_id=version_id)
    if len(deleted_models) == 1:
        return deleted_models[0]
    elif len(deleted_models) == 0:
        raise HTTPException(status_code=404, detail="Model version file not found")
    else:
        raise HTTPException(status_code=500, detail="Multiple model version files found")
