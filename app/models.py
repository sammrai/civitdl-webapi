import enum
from typing import Optional
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
    # Filled in once the Civitai metadata is fetched, so a failed task still
    # says which model it was. Civitai's own spelling, e.g. "LORA".
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    result: Optional[ModelInfo]
    error: Optional[str]