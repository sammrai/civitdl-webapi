from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.main import app
from app.models import ModelInfo
from unittest.mock import patch, ANY, MagicMock


client = TestClient(app)

mock_models = [
    ModelInfo(
        model_id=546949,
        version_id=1,
        model_dir="/path/to/models/model-mid_546949-vid_1.ckpt",
        filename="model-mid_546949-vid_1.ckpt",
        model_type="lora",
        name="Test Lora Model",
        description="A test lora model for unit testing",
        created_at="2023-01-01T00:00:00.000Z"
    ),
    ModelInfo(
        model_id=123456,
        version_id=3,
        model_dir="/path/to/models/model-mid_123456.pt",
        filename="model-mid_123456.pt",
        model_type="checkpoint",
        name="Test Checkpoint Model",
        description="A test checkpoint model for unit testing",
        created_at="2023-01-02T00:00:00.000Z"
    ),
]


@patch('app.routers.find_model_files')
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
            "name": "Test Lora Model",
            "description": "A test lora model for unit testing",
            "created_at": "2023-01-01T00:00:00.000Z"
        },
        {
            "model_id": 123456,
            "version_id": 3,
            "model_dir": "/path/to/models/model-mid_123456.pt",
            "filename": "model-mid_123456.pt",
            "model_type": "checkpoint",
            "name": "Test Checkpoint Model",
            "description": "A test checkpoint model for unit testing",
            "created_at": "2023-01-02T00:00:00.000Z"
        },
    ]
    mock_find_model_files.assert_called_once_with(model_id=None, version_id=None)

@patch('app.routers.find_model_files')
def test_get_model_success(mock_find_model_files):
    mock_find_model_files.return_value = [mock_models[0]]  # Return only the first model

    response = client.get("/models/546949")
    assert response.status_code == 200
    assert response.json() == [{
        "model_id": 546949,
        "version_id": 1,
        "model_dir": "/path/to/models/model-mid_546949-vid_1.ckpt",
        "filename": "model-mid_546949-vid_1.ckpt",
        "model_type": "lora",
        "name": "Test Lora Model",
        "description": "A test lora model for unit testing",
        "created_at": "2023-01-01T00:00:00.000Z"
    }]
    mock_find_model_files.assert_called_once_with(model_id=546949, version_id=None)

@patch('app.routers.find_model_files')
def test_get_model_not_found(mock_find_model_files):
    mock_find_model_files.return_value = mock_models

    response = client.get("/models/999999")
    assert response.status_code == 404
    assert response.json() == {"detail": "Model not found"}
    mock_find_model_files.assert_called_once_with(model_id=999999, version_id=None)

@patch('app.routers._civitdl')
def test_download_model_success(mock_civitdl):
    mock_civitdl.return_value = ModelInfo(
        model_id=546949,
        version_id=1,
        model_dir="folder1",
        model_type="lora",
        filename="example.safetensors",
        name="Downloaded Model",
        description="A model downloaded from Civitai",
        created_at="2023-01-03T00:00:00.000Z"
    )

    response = client.post("/models/546949/versions/1")
    assert response.status_code == 200
    assert response.json() == {
        "model_id": 546949,
        "version_id": 1,
        "model_dir": "folder1",
        "model_type": "lora",
        "filename": "example.safetensors",
        "name": "Downloaded Model",
        "description": "A model downloaded from Civitai",
        "created_at": "2023-01-03T00:00:00.000Z"
    }
    mock_civitdl.assert_called_once_with(model_id=546949, version_id=1, api_key=ANY)

@patch('app.routers._civitdl')
def test_download_model_failure(mock_civitdl):
    mock_civitdl.side_effect = HTTPException(status_code=304, detail="Model already downloaded.")

    response = client.post("/models/546949/versions/1")
    assert response.status_code == 304
    mock_civitdl.assert_called_once_with(model_id=546949, version_id=1, api_key=ANY)

@patch('app.routers.delete_model_files')
def test_remove_model_success(mock_delete_model_file):
    mock_delete_model_file.return_value = [
        ModelInfo(
            model_id=546949,
            version_id=1,
            model_dir="/path/to/models/model-mid_546949-vid_1.ckpt",
            filename="model-mid_546949-vid_1.ckpt",
            model_type="lora",
            name="Test Lora Model",
            description="A test lora model for unit testing",
            created_at="2023-01-01T00:00:00.000Z"
        )
    ]

    response = client.delete("/models/546949/versions/1")
    assert response.status_code == 200
    assert response.json() == {
            "model_id": 546949,
            "version_id": 1,
            "model_dir": "/path/to/models/model-mid_546949-vid_1.ckpt",
            "filename": "model-mid_546949-vid_1.ckpt",
            "model_type": "lora",
            "name": "Test Lora Model",
            "description": "A test lora model for unit testing",
            "created_at": "2023-01-01T00:00:00.000Z"
    }
    mock_delete_model_file.assert_called_once_with(model_id=546949, version_id=1)

@patch('app.routers.delete_model_files')
def test_remove_model_not_found(mock_delete_model_file):
    mock_delete_model_file.return_value = []

    response = client.delete("/models/999999/versions/1")
    assert response.status_code == 404
    mock_delete_model_file.assert_called_once_with(model_id=999999, version_id=1)

@patch('app.routers.delete_model_files')
def test_remove_all_models_success(delete_model_files):
    delete_model_files.return_value = [
        ModelInfo(
            model_id=546949,
            version_id=1,
            model_dir="/path/to/models/model-mid_546949-vid_1.ckpt",
            filename="model-mid_546949-vid_1.ckpt",
            model_type="lora",
            name="Test Lora Model",
            description="A test lora model for unit testing",
            created_at="2023-01-01T00:00:00.000Z"
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
            "name": "Test Lora Model",
            "description": "A test lora model for unit testing",
            "created_at": "2023-01-01T00:00:00.000Z"
        }
    ]
    delete_model_files.assert_called_once()


@patch('app.routers.find_model_files')
@patch('app.routers.os.path.exists')
@patch('app.routers.os.listdir')
@patch('app.routers.FileResponse')
def test_get_model_version_image_success(mock_file_response, mock_listdir, mock_exists, mock_find_model_files):
    mock_find_model_files.return_value = [mock_models[0]]
    mock_exists.return_value = True
    mock_listdir.return_value = ['image1.jpg', 'image2.png', 'notanimage.txt']
    mock_file_response.return_value = MagicMock()
    
    response = client.get("/models/546949/versions/1/image")
    # The actual response will be from FastAPI's test client, not our mock
    # So we just verify the function was called correctly
    mock_find_model_files.assert_called_once_with(model_id=546949, version_id=1)
    mock_file_response.assert_called_once()


@patch('app.routers.find_model_files')
def test_get_model_version_image_not_found(mock_find_model_files):
    mock_find_model_files.return_value = []
    
    response = client.get("/models/999999/versions/1/image")
    assert response.status_code == 404
    assert response.json() == {"detail": "Model version not found"}


@patch('app.routers.find_model_files')
@patch('app.routers.os.path.exists')
def test_get_model_version_image_no_directory(mock_exists, mock_find_model_files):
    mock_find_model_files.return_value = [mock_models[0]]
    mock_exists.return_value = False
    
    response = client.get("/models/546949/versions/1/image")
    assert response.status_code == 404
    assert response.json() == {"detail": "No images directory found"}


@patch('app.routers.find_model_files')
@patch('app.routers.os.path.exists')
@patch('app.routers.os.listdir')
def test_get_model_version_image_no_images(mock_listdir, mock_exists, mock_find_model_files):
    mock_find_model_files.return_value = [mock_models[0]]
    mock_exists.return_value = True
    mock_listdir.return_value = ['notanimage.txt', 'readme.md']
    
    response = client.get("/models/546949/versions/1/image")
    assert response.status_code == 404
    assert response.json() == {"detail": "No images found"}
