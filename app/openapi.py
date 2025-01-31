from fastapi.openapi.utils import get_openapi
from app.main import app
import json
import yaml
import argparse

def generate_openapi(output_file, output_format="yaml"):
    openapi_schema = get_openapi(
        title="CivitDL API",
        version="1.0.0",
        description="API for downloading Civitai models",
        routes=app.routes,
    )
    
    if output_format == "json":
        with open(output_file, "w") as f:
            json.dump(openapi_schema, f, indent=2)
    else:
        with open(output_file, "w") as f:
            yaml.dump(openapi_schema, f, sort_keys=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OpenAPI schema")
    parser.add_argument("output_file", type=str, help="Output file for the OpenAPI files")
    parser.add_argument("--format", type=str, choices=["json", "yaml"], default="yaml", help="Output format (json or yaml)")
    args = parser.parse_args()
    generate_openapi(args.output_file, args.format)