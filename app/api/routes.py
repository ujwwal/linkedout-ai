from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..models.schemas import PostRequest, PostResponse, PostChoice, ClientResponse
from ..db import crud
from ..db.database import get_db

router = APIRouter()

@router.post("/generate_post", response_model=PostResponse)
def generate_post(request: PostRequest, db: Session = Depends(get_db)):
    # Placeholder: Implement post generation logic
    # This will integrate with LangChain, FAISS, and Azure OpenAI in next steps
    return PostResponse(posts=[], user_type="beginner")

@router.post("/save_choice", response_model=bool)
def save_choice(choice: PostChoice, db: Session = Depends(get_db)):
    success = crud.save_post_choice(db, choice.post_id)
    if not success:
        raise HTTPException(status_code=404, detail="Post not found")
    return True

@router.get("/clients/{user_id}", response_model=ClientResponse)
def get_clients(user_id: str, db: Session = Depends(get_db)):
    clients = crud.get_clients(db, user_id)
    return ClientResponse(clients=clients)