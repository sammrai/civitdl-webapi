from app.main import app
from app.utils import MODEL_TYPE_TO_FOLDER

from fastapi import status
from fastapi.testclient import TestClient
import os
import pytest
import re
import tempfile


@pytest.fixture(scope="session")
def civitai_token():
    # Replace with a valid token or mock if necessary
    return os.getenv("CIVITAI_TOKEN", "")

@pytest.fixture(scope="session")
def model_root_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture(scope="function")  # Changed scope to 'function'
def client(model_root_path, civitai_token, monkeypatch):
    # Set environment variables for testing
    monkeypatch.setenv("MODEL_ROOT_PATH", model_root_path)
    monkeypatch.setenv("CIVITAI_TOKEN", civitai_token)
    
    with TestClient(app) as c:
        yield c

# Predefined model IDs and their types
MODEL_TEST_DATA = [
    {"model_id": 28205  , "type": "lora"            , "version_id": 47670},
    # {"model_id": 17   , "type": "checkpoint"      , "version_id": None},
    {"model_id": 4514   , "type": "textualinversion", "version_id": None},
    {"model_id": 79699  , "type": "vae"             , "version_id": None},
    {"model_id": 14878  , "type": "locon"           , "version_id": None},
]

# Regex pattern to match model filenames
MODEL_PATTERN = re.compile(r'.*-mid_(\d+)(?:-vid_(\d+))?.*\.(safetensors|ckpt|pt)$')

@pytest.mark.parametrize("model", MODEL_TEST_DATA)
def test_download_model(client, model):
    """
    Test downloading a model using the /models/{model_id}/versions/{version_id} endpoint.
    """
    model_id = model["model_id"]
    version_id = model["version_id"]

    url = f"/models/{model_id}"
    if version_id is not None:
        url += f"/versions/{version_id}"

    response = client.post(url)
    assert response.status_code == status.HTTP_200_OK, f"Failed to download model {model_id}"
    data = response.json()
    assert data["model_id"] == model_id
    if version_id is not None:
        assert data["version_id"] == version_id
    
    # Verify the file exists in the expected directory
    model_type = model["type"].lower()
    expected_folder = os.path.join(os.getenv("MODEL_ROOT_PATH", "/default/path"), MODEL_TYPE_TO_FOLDER.get(model_type, ""))
    assert os.path.exists(expected_folder), f"Expected folder {expected_folder} does not exist."
    
    # Find the downloaded file
    files = []
    for root, _, filenames in os.walk(expected_folder):
        for filename in filenames:
            if MODEL_PATTERN.match(filename):
                match = MODEL_PATTERN.match(filename)
                if int(match.group(1)) == model_id:
                    files.append(os.path.join(root, filename))
    
    assert len(files) > 0, f"Model file for {model_id} not found in {expected_folder}"

@pytest.mark.parametrize("model", MODEL_TEST_DATA)
def test_get_model(client, model):
    """
    Test retrieving a specific model's information using the /models/{model_id}/versions/{version_id} endpoint.
    """
    model_id = model["model_id"]
    version_id = model["version_id"]
    model_type = model["type"]

    url = f"/models/{model_id}"
    if version_id is not None:
        url += f"/versions/{version_id}"

    response = client.get(url)
    data = response.json()

    if version_id is None:
        print("#",data)
        data = data[0]
    
    if response.status_code == status.HTTP_200_OK:
        assert data["model_id"] == (model_id)
        assert data["model_type"] == (model_type)
        if version_id is not None:
            assert data["version_id"] == version_id
    else:
        pytest.fail(f"Model {model_id} not found when it should exist.")

def test_list_models(client):
    """
    Test listing all available models using the /models/ endpoint.
    """
    response = client.get("/models/")
    assert response.status_code == status.HTTP_200_OK, "Failed to list models"
    data = response.json()
    assert isinstance(data, list), "Response is not a list"
    # Verify that all test models are present
    model_ids = [(model["model_id"]) for model in MODEL_TEST_DATA]
    returned_ids = [model["model_id"] for model in data]
    for mid in model_ids:
        assert mid in returned_ids, f"Model ID {mid} not found in list"

@pytest.mark.parametrize("model", MODEL_TEST_DATA)
def test_delete_model(client, model):
    """
    Test deleting a specific model using the /models/{model_id}/versions/{version_id} endpoint.
    """
    model_id = model["model_id"]
    version_id = model["version_id"]

    url = f"/models/{model_id}"
    if version_id is not None:
        url += f"/versions/{version_id}"

    response = client.delete(url)
    data = response.json()

    if version_id is None:
        data = data[0]
    
    if response.status_code == status.HTTP_200_OK:
        assert data["model_id"] == model_id
        if version_id is not None:
            assert data["version_id"] == version_id
    else:
        pytest.fail(f"Failed to delete model {model_id}")
    
    # Verify the file has been deleted
    model_type = model["type"].lower()
    expected_folder = os.path.join(os.getenv("MODEL_ROOT_PATH", "/default/path"), MODEL_TYPE_TO_FOLDER.get(model_type, ""))
    assert os.path.exists(expected_folder), f"Expected folder {expected_folder} does not exist."
    
    # Check that no files exist for this model_id
    for root, _, filenames in os.walk(expected_folder):
        for filename in filenames:
            if MODEL_PATTERN.match(filename):
                print("##", filename)
                match = MODEL_PATTERN.match(filename)
                if int(match.group(1)) == model_id:
                    pytest.fail(f"Model file for {model_id} still exists after deletion.")

@pytest.mark.parametrize("model", [{"model_id": 28205, "type": "lora", "version_id": 33811}])
def test_delete_all_models(client, model):
    """
    Test deleting all models using the /models/ endpoint.
    """

    response = client.post(f"/models/{model['model_id']}/versions/{model['version_id']}")
    assert response.status_code == status.HTTP_200_OK

    response = client.post(f"/models/{model['model_id']}")
    num_models = len(response.json())

    response = client.delete("/models/")
    num = len(response.json())
    assert response.status_code == status.HTTP_200_OK
    
    # Verify all model files have been deleted
    for model in MODEL_TEST_DATA:
        model_id = model["model_id"]
        model_type = model["type"].lower()
        expected_folder = os.path.join(os.getenv("MODEL_ROOT_PATH", "/default/path"), MODEL_TYPE_TO_FOLDER.get(model_type, ""))
        
        for root, _, filenames in os.walk(expected_folder):
            for filename in filenames:
                if MODEL_PATTERN.match(filename):
                    match = MODEL_PATTERN.match(filename)
                    if int(match.group(1)) == model_id:
                        pytest.fail(f"Model file for {model_id} still exists after deleting all models.")

@pytest.mark.parametrize("model", [{"model_id": 28205, "type": "lora", "version_id": 33811}])
def test_download_and_get_multiple_versions(client, model):
    """
    Test downloading a model with multiple versions and verifying both versions are returned.
    """
    model_id = model["model_id"]
    version_ids = [33811, 47670]

    for version_id in version_ids:
        response = client.post(f"/models/{model_id}/versions/{version_id}")
        assert response.status_code == status.HTTP_200_OK, f"Failed to download model {model_id} version {version_id}"

    response = client.get(f"/models/{model_id}")
    assert response.status_code == status.HTTP_200_OK, f"Failed to get model {model_id}"
    data = response.json()
    returned_version_ids = [item["version_id"] for item in data]
    assert len(data) == len(version_ids)
    for version_id in version_ids:
        assert version_id in returned_version_ids, f"Version {version_id} not found in model {model_id}"

@pytest.mark.parametrize("model", [{"model_id": 28205, "type": "lora", "version_id": 33811}])
def test_delete_specific_version(client, model):
    """
    Test deleting a specific version of a model and verifying only that version is deleted.
    """
    response = client.post(f"/models/{model['model_id']}/versions/{model['version_id']}")

    model_id = model["model_id"]
    version_id_to_delete = 33811
    version_id_to_keep = 47670
    
    for version_id in [version_id_to_delete, version_id_to_keep]:
        response = client.post(f"/models/{model_id}/versions/{version_id}")
        print(response)
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_304_NOT_MODIFIED]

    response = client.get(f"/models/{model_id}")
    print(response.text)

    # Delete specific version
    response = client.delete(f"/models/{model_id}/versions/{version_id_to_delete}")
    assert response.status_code == status.HTTP_200_OK, f"Failed to delete model {model_id} version {version_id_to_delete}"

    # Verify only the specified version is deleted
    response = client.get(f"/models/{model_id}")
    print(response.text)
    assert response.status_code == status.HTTP_200_OK, f"Failed to get model {model_id}"
    data = response.json()
    returned_version_ids = [item["version_id"] for item in data]
    assert version_id_to_delete not in returned_version_ids, f"Version {version_id_to_delete} was not deleted"
    assert version_id_to_keep in returned_version_ids, f"Version {version_id_to_keep} was incorrectly deleted"

@pytest.mark.parametrize("model", [{"model_id": 28205, "type": "lora", "version_id": 33811}])
def test_delete_all_versions(client, model):
    """
    Test deleting all versions of a model and verifying all versions are deleted.
    """
    model_id = model["model_id"]
    version_ids = [33811, 47670]

    for version_id in version_ids:
        response = client.post(f"/models/{model_id}/versions/{version_id}")
    response = client.get(f"/models/")
    assert len(response.json())==2

    # Delete all versions
    response = client.delete(f"/models/{model_id}")
    assert response.status_code == status.HTTP_200_OK, f"Failed to delete model {model_id}"

    # Verify all versions are deleted
    response = client.get(f"/models/{model_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND, f"Model {model_id} was not fully deleted"

    response = client.get(f"/models/")
    assert len(response.json())==0


@pytest.mark.parametrize("model", [{"model_id": 28205, "type": "lora", "version_id": 33811}])
def test_download_same_version_twice(client, model):
    """
    Test downloading the same model version twice and verifying the second attempt returns a 304 status code.
    """
    model_id = model["model_id"]
    version_id = model["version_id"]

    # First download attempt
    response = client.post(f"/models/{model_id}/versions/{version_id}")
    assert response.status_code == status.HTTP_200_OK, f"Failed to download model {model_id} version {version_id}"

    # Second download attempt
    response = client.post(f"/models/{model_id}/versions/{version_id}")
    assert response.status_code == status.HTTP_304_NOT_MODIFIED, "Expected 304 status code for already downloaded model"

def test_get_nonexistent_model(client):
    """
    Test retrieving a nonexistent model and verifying it returns a 404 status code.
    """
    nonexistent_model_id = 999999

    response = client.get(f"/models/{nonexistent_model_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND, "Expected 404 status code for nonexistent model"

def test_delete_all_models_when_none_exist(client):
    """
    Test deleting all models when no models exist and verifying it returns a 404 status code.
    """
    # Ensure no models exist
    response = client.delete("/models/")

    # Attempt to delete all models again
    response = client.delete("/models/")
    assert response.status_code == status.HTTP_404_NOT_FOUND, "Expected 404 status code when no models exist"

def test_download_nonexistent_model(client):
    """
    Test downloading a nonexistent model and verifying it returns a 404 status code.
    """
    nonexistent_model_id = 9999999999999

    response = client.post(f"/models/{nonexistent_model_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND, "Expected 404 status code for nonexistent model"

def test_download_nonexistent_version(client):
    """
    Test downloading a model with a nonexistent version and verifying it returns a 404 status code.
    """
    model_id = 28205
    nonexistent_version_id = 1

    response = client.post(f"/models/{model_id}/versions/{nonexistent_version_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND, "Expected 404 status code for nonexistent version"
