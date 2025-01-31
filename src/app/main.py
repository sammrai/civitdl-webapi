from fastapi import FastAPI, Query, HTTPException, APIRouter
from app.routers import models_router
from app.routers import versions_router

app = FastAPI()

app.include_router(models_router)
app.include_router(versions_router)
