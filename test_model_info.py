#!/usr/bin/env python3
import json
import os

# Create a test model directory and metadata
test_dir = "/tmp/test_models/Stable-diffusion"
os.makedirs(test_dir, exist_ok=True)

# Create a test model file
model_file = os.path.join(test_dir, "test-model-mid_12345-vid_67890.safetensors")
with open(model_file, "w") as f:
    f.write("dummy model content")

# Create extra_data directory and JSON file
extra_data_dir = os.path.join(test_dir, "extra_data-vid_67890")
os.makedirs(extra_data_dir, exist_ok=True)

# Create model metadata JSON
metadata = {
    "type": "checkpoint",
    "name": "Test Model Name",
    "description": "This is a test model description for testing the API",
    "model_id": 12345,
    "version_id": 67890
}

json_file = os.path.join(extra_data_dir, "model_dict-mid_12345-vid_67890.json")
with open(json_file, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Created test model at: {model_file}")
print(f"Created metadata at: {json_file}")