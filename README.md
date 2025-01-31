# CivitDL Web API

**CivitDL Web API** is a RESTful wrapper for the original [CivitDL CLI tool](https://github.com/OwenTruong/civitdl), designed to simplify the downloading and management of models from [Civitai](https://civitai.com/). By using this API, you can automate and efficiently handle model downloads, making it particularly useful for projects like [sd-forge-docker](https://github.com/sammrai/sd-forge-docker) that integrate with **Stable Diffusion WebUI**.


## Installation

Follow these steps to install and set up the API:

1. **Clone the repository:**  
   ```sh
   git clone https://github.com/sammrai/civitdl-webapi.git
   cd civitdl-webapi
   ```

2. **Start the Docker container:**  
   ```sh
   docker-compose up -d
   ```

3. **Access the API:**  
   Open `http://localhost:7681` in your browser or use a tool like `curl` to interact with the API.


## Configuration

The API supports the following environment variables, which can be configured in `docker-compose.yml`:

| Variable          | Description                                        | Default Value |
|------------------|------------------------------------------------|--------------|
| `CIVITAI_TOKEN` | (Optional) Your Civitai API token for authentication. Required for downloading certain restricted models. | Not set |
| `MODEL_ROOT_PATH` | Directory where models will be stored.         | `/data/models` |

> **Note:** `CIVITAI_TOKEN` is not mandatory but is **highly recommended** for accessing models that require authentication. Without it, some models may not be downloadable.


## Model Directory Structure

Downloaded models are organized into directories based on their type. If `MODEL_ROOT_PATH=/data/models`, the structure will be:

- **LoRA models:** `/data/models/Lora`  
- **VAE models:** `/data/models/VAE`  
- **Checkpoint models:** `/data/models/Stable-diffusion`  
- **Textual Inversion models:** `/data/models/text_encoder`  

This structure ensures efficient model organization and easy management.


## API Endpoints

For detailed OpenAPI specifications, please refer to the following links:
- [Redoc](https://sammrai.github.io/civitdl-webapi/redoc.html)
- [Swagger](https://sammrai.github.io/civitdl-webapi/swagger.html)
- [Stoplight](https://sammrai.github.io/civitdl-webapi/stoplight.html)

### Download a model by ID
Downloads a specific model from Civitai.  

#### Request:
```sh
curl -s -X POST "http://localhost:7681/models/30410"
```

#### Response:
```json
{
  "model_id": 30410,
  "version_id": 93602,
  "model_dir": "/data/Lora/Pokemon - Selene-mid_30410-vid_93602",
  "filename": "Selene-10-mid_30410-vid_93602.safetensors",
  "model_type": "lora"
}
```

### Download a model by model ID and version ID
Downloads a specific model from Civitai.  

#### Request:
```sh
curl -s -X POST "http://localhost:7681/models/30410/versions/36664"
```

#### Response:
```json
{
  "model_id": 30410,
  "version_id": 36664,
  "model_dir": "/data/Lora/Pokemon - Selene-mid_30410-vid_36664",
  "filename": "SeleneLora-10-mid_30410-vid_36664.safetensors",
  "model_type": "lora"
}
```


### List all downloaded models
Retrieves a list of all models stored in the system.  

#### Request:
```sh
curl -X GET "http://localhost:7681/models/"
```

#### Response:
```json
[
  {
    "model_id": 30410,
    "version_id": 93602,
    "model_dir": "/data/Lora/Pokemon - Selene-mid_30410-vid_93602",
    "filename": "Selene-10-mid_30410-vid_93602.safetensors",
    "model_type": "lora"
  },
  {
    "model_id": 30410,
    "version_id": 36664,
    "model_dir": "/data/Lora/Pokemon - Selene-mid_30410-vid_36664",
    "filename": "SeleneLora-10-mid_30410-vid_36664.safetensors",
    "model_type": "lora"
  }
]
```


## Persistent Storage

The following volume is defined in `docker-compose.yml` to persist downloaded models:

- `./data:/data` – Maps the local `data` directory to the container’s `/data` directory, ensuring models remain available even if the container is restarted.


## Ports

By default, the API is exposed on port `7681`. The mapping in `docker-compose.yml` is:

- **Host:** `7681` → **Container:** `7681`

You can modify this if needed.


## Acknowledgments

This project is powered by [CivitDL](https://github.com/OwenTruong/civitdl).  
