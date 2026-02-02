from typing import List
import os
import threading

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from app.models import ModelInfo, AsyncDownloadResponse, TaskStatus
from app.utils import (
    _civitdl,
    _civitdl_async_worker,
    create_task,
    get_task,
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

status_router = APIRouter(
    prefix="/status",
    tags=["status"],
)


# --- Endpoints ---
@models_router.get("/", response_model=List[ModelInfo])
def list_all_models():
    """
    Retrieve a list of all saved models.
    This endpoint fetches all model files available in the system and returns their information.
    """
    models = find_model_files(model_id=None, version_id=None)
    return [ModelInfo(**model.__dict__) for model in models]


@models_router.get("/{model_id}", response_model=List[ModelInfo])
def list_model_versions(model_id: int):
    """
    Retrieve a list of all saved versions for the specified model ID.

    This endpoint fetches all versions of a specific model available in the system.
    """
    models = find_model_files(model_id=model_id, version_id=None)
    if not models:
        raise HTTPException(status_code=404, detail="Model not found")
    return [ModelInfo(**model.__dict__) for model in models]


@models_router.post("/{model_id}", response_model=ModelInfo)
def download_model(model_id: int):
    """
    Download the latest version of the specified model.

    This endpoint initiates the download process for the latest version of the given model.
    """
    return _civitdl(model_id=model_id, version_id=None, api_key=CIVITAI_TOKEN)


@models_router.delete("/{model_id}", response_model=List[ModelInfo])
def remove_model(model_id: int):
    """
    Delete all versions of the specified model ID.

    This endpoint removes all files associated with the given model ID from the system.
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
    """
    return _civitdl(model_id=model_id, version_id=version_id, api_key=CIVITAI_TOKEN)


@versions_router.delete("/{version_id}", response_model=ModelInfo)
def remove_model_version(model_id: int, version_id: int):
    """
    Delete the specified version of the model.

    This endpoint removes a specific version of a model from the system.
    """
    deleted_models = delete_model_files(model_id=model_id, version_id=version_id)
    if len(deleted_models) == 1:
        return deleted_models[0]
    elif len(deleted_models) == 0:
        raise HTTPException(status_code=404, detail="Model version file not found")
    else:
        raise HTTPException(status_code=500, detail="Multiple model version files found")


@versions_router.get("/{version_id}/image")
def get_model_version_image(model_id: int, version_id: int):
    """
    Get the first image for the specified model version.

    This endpoint returns the first image found in the model's extra_data directory.
    """
    models = find_model_files(model_id=model_id, version_id=version_id)
    if not models:
        raise HTTPException(status_code=404, detail="Model version not found")

    model = models[0]
    # Look for images in the extra_data directory
    extra_data_dir = os.path.join(model.model_dir, f"extra_data-vid_{version_id}")

    if not os.path.exists(extra_data_dir):
        raise HTTPException(status_code=404, detail="No images directory found")

    # Find image files using os.listdir to handle special characters
    image_files = [
        os.path.join(extra_data_dir, file)
        for file in os.listdir(extra_data_dir)
        if file.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not image_files:
        raise HTTPException(status_code=404, detail="No images found")

    # Return the first image
    image_path = sorted(image_files)[0]  # Sort to ensure consistent ordering
    return FileResponse(image_path)


# --- Async Download Endpoints ---
@models_router.post("/{model_id}/async", response_model=AsyncDownloadResponse)
def download_model_async(model_id: int, background_tasks: BackgroundTasks):
    """
    Start an asynchronous download of the latest version of the specified model.

    This endpoint initiates a background download and returns a task ID for status tracking.
    """
    task_id = create_task(model_id=model_id, version_id=None)

    # Start background download using threading to avoid blocking
    thread = threading.Thread(
        target=_civitdl_async_worker,
        args=(task_id, model_id, None, CIVITAI_TOKEN)
    )
    thread.daemon = True
    thread.start()

    return AsyncDownloadResponse(
        task_id=task_id,
        status_url=f"/status/{task_id}"
    )


@versions_router.post("/{version_id}/async", response_model=AsyncDownloadResponse)
def download_model_version_async(model_id: int, version_id: int, background_tasks: BackgroundTasks):
    """
    Start an asynchronous download of the specified version of the model.

    This endpoint initiates a background download and returns a task ID for status tracking.
    """
    task_id = create_task(model_id=model_id, version_id=version_id)

    # Start background download using threading to avoid blocking
    thread = threading.Thread(
        target=_civitdl_async_worker,
        args=(task_id, model_id, version_id, CIVITAI_TOKEN)
    )
    thread.daemon = True
    thread.start()

    return AsyncDownloadResponse(
        task_id=task_id,
        status_url=f"/status/{task_id}"
    )


@status_router.get("/{task_id}", response_model=TaskStatus)
def get_download_status(task_id: str):
    """
    Get the status and progress of an asynchronous download task.

    Returns the current status (pending, downloading, finished, failed),
    progress percentage (0-100), and result or error information.
    """
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskStatus(**task)