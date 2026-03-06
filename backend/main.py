from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.deps import get_rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_rag()  # warm up on startup
    yield


app = FastAPI(title="DeepChatLocal API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
