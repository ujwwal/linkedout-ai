from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.api.routes import router
from app.db import models
from app.db.database import engine

# Create tables if they don't exist
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="LinkedIn Post Generator API")

# Add CORS middleware to allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(router, tags=["posts"])

@app.get("/")
def root():
    return {"message": "Welcome to LinkedIn Post Generator API"}