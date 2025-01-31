from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.main import app, ModelInfo
from unittest.mock import patch
from unittest.mock import patch, ANY


client = TestClient(app)

mock_models = [
    ModelInfo(
        model_id=546949,
        version_id=1,
        model_dir="/path/to/models/model-mid_546949-vid_1.ckpt",
        filename="model-mid_546949-vid_1.ckpt",
        model_type="lora",
    ),
    ModelInfo(
        model_id=123456,
        version_id=3,
        model_dir="/path/to/models/model-mid_123456.pt",
        filename="model-mid_123456.pt",
        model_type="checkpoint",
    ),
]


@patch('app.main.find_model_files')
def test_list_models(mock_find_model_files):
    mock_find_model_files.return_value = mock_models

    response = client.get("/models/")
    assert response.status_code == 200
    assert response.json() == [
        {
            "model_id": 546949,
            "version_id": 1,
            "model_dir": "/path/to/models/model-mid_546949-vid_1.ckpt",
            "filename": "model-mid_546949-vid_1.ckpt",
            "model_type": "lora",
        },
        {
            "model_id": 123456,
            "version_id": 3,
            "model_dir": "/path/to/models/model-mid_123456.pt",
            "filename": "model-mid_123456.pt",
            "model_type": "checkpoint",

        },
    ]
    mock_find_model_files.assert_called_once_with(None, None)

@patch('app.main.find_model_files')
def test_get_model_success(mock_find_model_files):
    mock_find_model_files.return_value = mock_models

    response = client.get("/models/546949")
    assert response.status_code == 200
    assert response.json() == [{
        "model_id": 546949,
        "version_id": 1,
        "model_dir": "/path/to/models/model-mid_546949-vid_1.ckpt",
        "filename": "model-mid_546949-vid_1.ckpt",
        "model_type": "lora",
    }]
    mock_find_model_files.assert_called_once_with(None, None)

@patch('app.main.find_model_files')
def test_get_model_not_found(mock_find_model_files):
    mock_find_model_files.return_value = mock_models

    response = client.get("/models/999999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Model not found"}
    mock_find_model_files.assert_called_once_with(None, None)

@patch('app.main._civitdl')
def test_download_model_success(mock_civitdl):
    mock_civitdl.return_value = {
        "model_id": 546949,
        "version_id": 1,
        "model_dir": "folder1",
        "model_type": "lora",
    }

    response = client.post("/models/546949/versions/1")
    assert response.status_code == 200
    assert response.json() == {
        "model_id": 546949,
        "version_id": 1,
        "model_dir": "folder1",
        "model_type": "lora",
    }
    mock_civitdl.assert_called_once_with(546949, version_id=1, api_key=ANY)

@patch('app.main._civitdl')
def test_download_model_failure(mock_civitdl):
    mock_civitdl.side_effect = HTTPException(status_code=304, detail="Model already downloaded.")

    response = client.post("/models/546949/versions/1")
    assert response.status_code == 304
    mock_civitdl.assert_called_once_with(546949, version_id=1, api_key=ANY)

@patch('app.main.delete_model_files')
def test_remove_model_success(mock_delete_model_file):
    mock_delete_model_file.return_value = [
        ModelInfo(
            model_id=546949,
            version_id=1,
            model_dir="/path/to/models/model-mid_546949-vid_1.ckpt",
            filename="model-mid_546949-vid_1.ckpt",
            model_type="lora",
        )
    ]

    response = client.delete("/models/546949/versions/1")
    assert response.status_code == 200
    assert response.json() == [
        {
            "model_id": 546949,
            "version_id": 1,
            "model_dir": "/path/to/models/model-mid_546949-vid_1.ckpt",
            "filename": "model-mid_546949-vid_1.ckpt",
            "model_type": "lora",
        }
    ]
    mock_delete_model_file.assert_called_once_with(546949, 1)

@patch('app.main.delete_model_files')
def test_remove_model_not_found(mock_delete_model_file):
    mock_delete_model_file.return_value = False

    response = client.delete("/models/999999/versions/1")
    assert response.status_code == 404
    mock_delete_model_file.assert_called_once_with(999999, 1)


@patch('app.main.delete_all_models')
def test_remove_all_models_success(mock_delete_all_model_files):
    mock_delete_all_model_files.return_value = [
        ModelInfo(
            model_id=546949,
            version_id=1,
            model_dir="/path/to/models/model-mid_546949-vid_1.ckpt",
            filename="model-mid_546949-vid_1.ckpt",
            model_type="lora",
        )
    ]

    response = client.delete("/models/")
    assert response.json() == [
        {
            "model_id": 546949,
            "version_id": 1,
            "model_dir": "/path/to/models/model-mid_546949-vid_1.ckpt",
            "filename": "model-mid_546949-vid_1.ckpt",
            "model_type": "lora",
        }
    ]
    mock_delete_all_model_files.assert_called_once()
