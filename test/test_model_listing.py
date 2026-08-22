"""Listing models found on disk, including ones with missing metadata."""

import json
import os

from unittest.mock import patch

import pytest

from fastapi.testclient import TestClient

from app import utils
from app.main import app


def write_model(root, type_dir, name, model_id, version_id, model_type="LORA", metadata=True):
    """Lay out a model the way civitdl's sorter does."""
    model_dir = os.path.join(root, type_dir, f"{name}-mid_{model_id}-vid_{version_id}")
    os.makedirs(model_dir, exist_ok=True)
    filename = f"{name}-mid_{model_id}-vid_{version_id}.safetensors"
    with open(os.path.join(model_dir, filename), "wb") as file:
        file.write(b"weights")

    if metadata:
        extra = os.path.join(model_dir, f"extra_data-vid_{version_id}")
        os.makedirs(extra, exist_ok=True)
        path = os.path.join(extra, f"model_dict-mid_{model_id}-vid_{version_id}.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "type": model_type,
                    "name": name,
                    "description": "",
                    "modelVersions": [{"id": version_id, "createdAt": "2026-01-01T00:00:00Z"}],
                },
                file,
            )
    return model_dir


@pytest.fixture
def model_root(tmp_path):
    with patch.object(utils, "MODEL_ROOT_PATH", str(tmp_path)):
        yield str(tmp_path)


def test_a_model_without_metadata_is_listed_as_unknown(model_root):
    write_model(model_root, "models/Lora", "Complete", 1, 10)
    write_model(model_root, "models/Lora", "Orphan", 2, 20, metadata=False)

    found = {model.model_id: model for model in utils.find_model_files()}

    assert found[1].model_type.value == "lora"
    assert found[2].model_type.value == "unknown"


def test_one_unreadable_model_does_not_break_the_listing(model_root):
    # A stray model file used to fail ModelInfo validation and 500 the whole
    # endpoint, hiding every other model.
    write_model(model_root, "models/Lora", "Complete", 1, 10)
    write_model(model_root, "models/Lora", "Orphan", 2, 20, metadata=False)

    response = TestClient(app).get("/models/")

    assert response.status_code == 200
    assert {model["model_id"] for model in response.json()} == {1, 2}


def test_progress_counts_only_this_models_tmp(model_root):
    """Two downloads of the same type must not report each other's bytes."""
    mine = write_model(model_root, "models/Lora", "Mine", 1, 10)
    other = write_model(model_root, "models/Lora", "Other", 2, 20)

    for directory, size in ((mine, 300), (other, 999)):
        tmp = os.path.join(directory, ".tmp")
        os.makedirs(tmp, exist_ok=True)
        with open(os.path.join(tmp, "part.safetensors"), "wb") as file:
            file.write(b"x" * size)

    metadata = {
        "model_dict": {"id": 1, "name": "Mine"},
        "version_dict": {"id": 10},
    }
    model_dir = utils._model_dir(metadata, os.path.join(model_root, "models/Lora"))

    assert utils._get_tmp_file_size(model_dir) == 300
    # The old behaviour: everything of that type lumped together.
    assert utils._get_tmp_file_size(os.path.join(model_root, "models/Lora")) == 1299
