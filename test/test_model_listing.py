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


def write_full_metadata(root, type_dir, name, model_id, version_id):
    """A model whose extra_data carries everything Civitai actually returns."""
    model_dir = write_model(root, type_dir, name, model_id, version_id)
    path = os.path.join(
        model_dir,
        f"extra_data-vid_{version_id}",
        f"model_dict-mid_{model_id}-vid_{version_id}.json",
    )
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "type": "LORA",
                "name": name,
                "description": "<p>model blurb</p>",
                "nsfw": False,
                "nsfwLevel": 3,
                "tags": ["style", "anime"],
                "creator": {"username": "someone"},
                "modelVersions": [
                    # An older version, to prove the fields are read off the
                    # version that is on disk and not off the first one.
                    {
                        "id": version_id - 1,
                        "name": "v1",
                        "baseModel": "SD 1.5",
                        "createdAt": "2025-01-01T00:00:00Z",
                    },
                    {
                        "id": version_id,
                        "name": "v2",
                        "baseModel": "Illustrious",
                        "baseModelType": "Standard",
                        "createdAt": "2026-01-01T00:00:00Z",
                        "publishedAt": "2026-01-02T00:00:00Z",
                        "description": "<p>version notes</p>",
                        "trainedWords": ["various colors"],
                        "stats": {"downloadCount": 2062, "thumbsUpCount": 398},
                        "files": [
                            {"name": "extra.vae", "sizeKB": 1.0, "primary": False},
                            {
                                "name": "weights.safetensors",
                                "sizeKB": 37199.25,
                                "primary": True,
                                "hashes": {"AutoV2": "2fcd88e6", "SHA256": "ABC123"},
                            },
                        ],
                    },
                ],
            },
            file,
        )
    return model_dir


def test_listing_reports_the_metadata_saved_next_to_the_model(model_root):
    write_full_metadata(model_root, "models/Lora", "Rainbow", 1, 10)

    model = utils.find_model_files(model_id=1, version_id=10)[0]

    assert model.base_model == "Illustrious"
    assert model.base_model_type == "Standard"
    assert model.version_name == "v2"
    assert model.version_description == "<p>version notes</p>"
    assert model.published_at == "2026-01-02T00:00:00Z"
    assert model.trained_words == ["various colors"]
    assert model.tags == ["style", "anime"]
    assert model.creator == "someone"
    assert model.nsfw is False
    assert model.nsfw_level == 3
    assert model.download_count == 2062
    assert model.thumbs_up_count == 398
    # Of the primary file, not of the first one listed.
    assert model.file_size_kb == 37199.25
    assert model.sha256 == "ABC123"


def test_the_endpoint_serves_the_extra_fields(model_root):
    write_full_metadata(model_root, "models/Lora", "Rainbow", 1, 10)

    response = TestClient(app).get("/models/")

    assert response.status_code == 200
    assert response.json()[0]["base_model"] == "Illustrious"


def test_missing_metadata_leaves_the_extra_fields_empty(model_root):
    """No extra_data at all must still list, the way model_type does."""
    write_model(model_root, "models/Lora", "Orphan", 2, 20, metadata=False)

    model = utils.find_model_files(model_id=2)[0]

    assert model.base_model is None
    assert model.trained_words == []
    assert model.tags == []


def test_a_version_absent_from_the_metadata_keeps_the_model_level_fields(model_root):
    """extra_data that does not describe this version id still names the model."""
    write_full_metadata(model_root, "models/Lora", "Rainbow", 1, 10)
    path = os.path.join(
        model_root,
        "models/Lora",
        "Rainbow-mid_1-vid_10",
        "extra_data-vid_10",
        "model_dict-mid_1-vid_10.json",
    )
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    data["modelVersions"] = [version for version in data["modelVersions"] if version["id"] != 10]
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file)

    model = utils.find_model_files(model_id=1)[0]

    assert model.name == "Rainbow"
    assert model.tags == ["style", "anime"]
    assert model.base_model is None
    assert model.created_at == ""
