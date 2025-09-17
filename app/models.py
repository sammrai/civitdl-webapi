import enum
from civitdl.batch._model import Model
from pydantic import BaseModel


class ModelType(enum.Enum):
    LORA = "lora"
    LOCON = "locon"
    DORA = "dora"
    VAE = "vae"
    CHECKPOINT = "checkpoint"
    TEXTUALINVERSION = "textualinversion"

# --- Data Models ---
class ModelInfo(BaseModel):
    model_id: int
    version_id: int
    model_dir: str
    filename: str
    model_type: ModelType  # Enum に変更
    name: str
    description: str
    created_at: str

class DownloadResponse(BaseModel):
    model_id: int
    version_id: int
    model_dir: str
    model_type: ModelType  # Enum に変更