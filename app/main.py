"""Application entry point and FastAPI factory."""
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from prometheus_client import make_asgi_app

from app.api.routes import router as api_router
from app.api.admin import admin_router
from app.logger import get_logger
from app.config import settings
from app.models.client import close_http_client

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AdaptiveFlow starting up — adapts to every prompt")
    try:
        yield
    finally:
        await close_http_client()
        logger.info("Shutting down")


app = FastAPI(
    title="AdaptiveFlow",
    description="AdaptiveFlow — adapts to every prompt. Production-grade LLM load balancer with RL routing",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-Key", "X-User-Id"],
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Register API routes
app.include_router(api_router)
app.include_router(admin_router)

# Serve static UI
app.mount("/static", StaticFiles(directory="ui/static"), name="static")


@app.get("/", include_in_schema=False)
async def root_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui")


@app.get("/ui", include_in_schema=False)
async def serve_ui():
    return FileResponse("ui/index.html")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
