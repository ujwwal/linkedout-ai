from fastapi import FastAPI
from app.api.routes import router
from app.db import models
from app.db.database import engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="LinkedIn Post Generator API")
app.include_router(router, tags=["posts"])

@app.get("/")
def root():
    return {"message": "Welcome to LinkedIn Post Generator API"}