"""Code paths the rest of the suite never reaches.

Measured by tracing which functions in app/ the unit tests call; these were the
ones that never ran. The endpoint ones were only exercised by the integration
suite, which CI skips.
"""

import json
import os
import sys

from unittest.mock import MagicMock, patch

import pytest

from fastapi.testclient import TestClient

from app import utils
from app.main import app
from app.models import ModelInfo

MODEL = ModelInfo(
    model_id=16014,
    version_id=28907,
    model_dir="/data/models/Lora/Anime Lineart-mid_16014-vid_28907",
    filename="animeoutlineV4_16-mid_16014-vid_28907.safetensors",
    model_type="lora",
    name="Anime Lineart",
    description="",
    created_at="2026-01-01T00:00:00Z",
)


@pytest.fixture
def client():
    return TestClient(app)


# --- endpoints only the integration suite touched ---------------------------

@patch("app.routers._civitdl")
def test_downloading_a_model_without_a_version(mock_civitdl, client):
    mock_civitdl.return_value = MODEL

    response = client.post("/models/16014")

    assert response.status_code == 200
    assert response.json()["filename"] == MODEL.filename
    assert mock_civitdl.call_args.kwargs["version_id"] is None


@patch("app.routers.delete_model_files")
def test_deleting_every_version_of_a_model(mock_delete, client):
    mock_delete.return_value = [MODEL]

    response = client.delete("/models/16014")

    assert response.status_code == 200
    assert len(response.json()) == 1
    mock_delete.assert_called_once_with(model_id=16014, version_id=None)


@patch("app.routers.find_model_files")
def test_getting_one_version(mock_find, client):
    mock_find.return_value = [MODEL]

    response = client.get("/models/16014/versions/28907")

    assert response.status_code == 200
    assert response.json()["version_id"] == 28907


@patch("app.routers.find_model_files")
def test_getting_a_version_that_is_not_there(mock_find, client):
    mock_find.return_value = []

    assert client.get("/models/16014/versions/28907").status_code == 404


# --- utils nothing else calls ------------------------------------------------

def test_wrap_cli_args_restores_argv():
    """It swaps sys.argv out under the CLI parser; a leak breaks every later call."""
    before = list(sys.argv)

    result = utils.wrap_cli_args(
        lambda: {"seen": list(sys.argv), "api_key": None},
        ["123", "/data"],
        api_key="secret",
    )

    assert sys.argv == before
    assert result["seen"] == ["cli_tool", "123", "/data"]
    assert result["api_key"] == "secret"


def test_wrap_cli_args_restores_argv_when_the_parser_raises():
    before = list(sys.argv)

    def explode():
        raise SystemExit(2)

    with pytest.raises(SystemExit):
        utils.wrap_cli_args(explode, ["--bad"])

    assert sys.argv == before


def test_wrap_cli_args_ignores_overrides_the_parser_does_not_know():
    result = utils.wrap_cli_args(lambda: {"known": 1}, [], known=2, unknown=3)

    assert result == {"known": 2}


def test_delete_removes_the_whole_model_directory(tmp_path):
    from test.test_model_listing import write_model

    with patch.object(utils, "MODEL_ROOT_PATH", str(tmp_path)):
        keep = write_model(str(tmp_path), "models/Lora", "Keep", 1, 10)
        drop = write_model(str(tmp_path), "models/Lora", "Drop", 2, 20)

        deleted = utils.delete_model_files(2, 20)

    assert [model.model_id for model in deleted] == [2]
    assert not os.path.exists(drop)
    assert os.path.exists(keep)


def test_deleting_something_that_is_not_there_is_not_an_error(tmp_path):
    with patch.object(utils, "MODEL_ROOT_PATH", str(tmp_path)):
        assert utils.delete_model_files(99, 99) == []


# --- the spec the docs job publishes -----------------------------------------

def test_the_openapi_spec_can_be_generated(tmp_path):
    from app import openapi

    out = tmp_path / "openapi.json"
    openapi.generate_openapi(str(out), "json")

    spec = json.loads(out.read_text(encoding="utf-8"))
    assert "/models/" in spec["paths"]
    assert "/status/{task_id}" in spec["paths"]

    yaml_out = tmp_path / "openapi.yaml"
    openapi.generate_openapi(str(yaml_out))
    assert yaml_out.read_text(encoding="utf-8").startswith("openapi:")
