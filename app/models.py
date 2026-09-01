import enum
from typing import List, Optional
from civitdl.batch._model import Model
from pydantic import BaseModel


class ModelType(enum.Enum):
    LORA = "lora"
    LOCON = "locon"
    DORA = "dora"
    VAE = "vae"
    CHECKPOINT = "checkpoint"
    TEXTUALINVERSION = "textualinversion"
    # A model file whose extra_data metadata is missing. Without this the whole
    # listing fails validation because of one unreadable directory.
    UNKNOWN = "unknown"

# --- Data Models ---
class ModelInfo(BaseModel):
    model_id: int
    version_id: int
    model_dir: str
    filename: str
    model_type: ModelType  # Enum に変更
    name: Optional[str]
    description: Optional[str]
    created_at: Optional[str]

    # Everything below comes from the same extra_data JSON civitdl already
    # writes next to the model file -- no extra call to Civitai. All optional:
    # a directory whose extra_data is missing must still list (see ModelType
    # .UNKNOWN), and older downloads predate some of these keys.

    # Which checkpoint family the version was trained against, e.g. "SD 1.5",
    # "Pony", "Illustrious". A plain str, not an enum: Civitai adds new base
    # models regularly and an unknown one must not fail the whole listing.
    base_model: Optional[str] = None
    base_model_type: Optional[str] = None  # e.g. "Standard", "Inpainting"
    # The version's own name ("LineV1"), as opposed to `name`, the model's.
    version_name: Optional[str] = None
    version_description: Optional[str] = None
    published_at: Optional[str] = None
    # Activation keywords. Empty for most checkpoints, essential for a LoRA.
    trained_words: List[str] = []
    tags: List[str] = []
    creator: Optional[str] = None
    nsfw: Optional[bool] = None
    nsfw_level: Optional[int] = None
    download_count: Optional[int] = None
    thumbs_up_count: Optional[int] = None
    # Of the version's primary file -- the only one civitdl downloads.
    file_size_kb: Optional[float] = None
    sha256: Optional[str] = None

class DownloadResponse(BaseModel):
    model_id: int
    version_id: int
    model_dir: str
    model_type: ModelType  # Enum に変更

class AsyncDownloadResponse(BaseModel):
    task_id: str
    status_url: str

class TaskStatus(BaseModel):
    task_id: str
    status: str  # pending, downloading, finished, failed
    progress: int  # 0-100
    model_id: int
    version_id: Optional[int]
    # Which model the task is for, so a failed task still says what it was
    # downloading. Same spelling as ModelInfo.model_type, e.g. "lora"; a plain
    # str because Civitai has types this API has no enum member for.
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    result: Optional[ModelInfo]
    error: Optional[str]