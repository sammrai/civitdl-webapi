# civitdl-webapi

A FastAPI wrapper around the [civitdl](https://github.com/OwenTruong/civitdl) CLI
for downloading Civitai models into a Stable Diffusion WebUI layout.

## Running the tests

The code needs `civitdl`, which is only installed inside the image. Mount the
source instead of rebuilding — a rebuild per edit is slow and unnecessary:

```sh
docker run --rm -v "$PWD":/app -e PYTHONPATH=/app <image> pytest -m "not integration"
```

`-m "not integration"` is the scope CI uses. Without it you also run
`test/test_integration_main.py`, which downloads real models from civitai.com.

If the container is already up: `docker-compose exec python-dev pytest test/`.

## Civitai rate limiting

Running the integration suite repeatedly gets the host 429'd:

```sh
curl -o /dev/null -w '%{http_code}' https://civitai.com/api/v1/models/28205   # 429
```

Once that happens, unrelated tests fail too, and it takes minutes to clear. This
is why CI skips the suite — every run used to fail on two integration tests that
had nothing to do with the change. If tests start failing in ways that make no
sense, check for a 429 before debugging the code.

## Traps

**Mocking `find_model_files` hides bugs in it.** Every listing test patches it
and feeds the endpoint `ModelInfo` objects that are valid by construction. A
real bug lived there for a long time — it emitted `model_type="unknown"`, which
was not in the `ModelType` enum, so `GET /models/` returned 500 for *every*
model whenever one directory was missing its `extra_data`. Tests that build the
directory layout on disk and go through the real function are in
`test/test_model_listing.py`.

**`MODEL_ROOT_PATH` is read at import time.** `monkeypatch.setenv` after import
does nothing; `test_integration_main.py` sets it and still writes to the real
`/data`. Patch the module attribute instead: `patch.object(utils,
"MODEL_ROOT_PATH", tmp_path)`.

**civitdl swallows the reason a download failed.** It reports every 401 as
"requires a valid API Key" and raises `APIException` with the real status on it.
Read `APIException.status_code`, and read the message off the response through
`_CivitaiSession`, which keeps the body of a 401/403 from the request civitdl
already made. Do not issue a second request to Civitai to find out why.

**civitdl can fail without raising.** Its retry loop catches the exception and
returns normally, leaving no model file. Treat "finished but wrote nothing" as a
failure, not as success.

## Single worker, on purpose

The Dockerfile runs uvicorn with `--workers 1`. Async download tasks live in a
process-local dict, so extra workers each get their own copy and
`GET /status/{task_id}` 404s whenever it lands on the wrong one. Raising the
worker count requires moving that state out of process first.

For the same reason the duplicate-download lock is a `threading.Lock`: it only
holds within one process. Task state is also lost on restart. Both are known and
accepted at this scale.

## Layout

| | |
|---|---|
| `app/routers.py` | HTTP endpoints. All plain `def`, so FastAPI runs them in its threadpool |
| `app/utils.py` | Downloads, task state, filesystem lookups |
| `app/sorter.py` | Where a model version lands on disk. The single source of that naming |
| `app/models.py` | Pydantic models |
