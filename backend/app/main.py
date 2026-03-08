from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from app.api import diagnosis, health  # health must be imported

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes FIRST — before any static file handling
app.include_router(health.router, tags=["Health"])
app.include_router(diagnosis.router, prefix="/api/v1", tags=["Diagnosis"])

# Static file serving LAST
react_build = Path("/app/frontend/build")
if react_build.exists():
    app.mount("/static", StaticFiles(directory=str(react_build / "static")), name="static")

    @app.get("/")
    def serve_root():
        return FileResponse(str(react_build / "index.html"))

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        # Let API routes through
        if full_path.startswith("api/") or full_path == "health":
            return {"error": "not found"}
        file_path = react_build / full_path
        if file_path.exists():
            return FileResponse(str(file_path))
        return FileResponse(str(react_build / "index.html"))
else:
    @app.get("/")
    def root():
        return {"message": "COVID-19 CDSS API running", "docs": "/docs"}
