from app.routers import models_router
from app.routers import versions_router
from app.routers import status_router

from fastapi import FastAPI


app = FastAPI()

app.include_router(models_router)
app.include_router(versions_router)
app.include_router(status_router)
