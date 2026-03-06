from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from api.deps import get_rag
from utils import log

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting DeepChat backend...")
    get_rag()
    log.info("RAG pipeline ready.")
    yield
    log.info("Shutting down.")

app = FastAPI(title="DeepChatLocal API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - t0
    log.info(f"HTTP {request.method} {request.url.path} → {response.status_code} ({elapsed:.2f}s)")
    response.headers["X-Process-Time"] = f"{elapsed:.3f}"
    return response


app.include_router(router, prefix="/api/v1")
