from fastapi.openapi.utils import get_openapi
from main import app
import json
import yaml
import argparse
import os

def generate_openapi(output_file):
    openapi_schema = get_openapi(
        title="CivitDL API",
        version="1.0.0",
        description="API for downloading Civitai models",
        routes=app.routes,
    )
    
    # YAML形式で保存
    with open(output_file, "w") as f:
        yaml.dump(openapi_schema, f, sort_keys=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OpenAPI schema")
    parser.add_argument("output_file", type=str, help="Output file for the OpenAPI files")
    args = parser.parse_args()
    generate_openapi(args.output_file)