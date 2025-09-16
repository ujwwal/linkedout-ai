from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..models.schemas import PostRequest, PostResponse, PostChoice, ClientResponse, GeneratedPost, UserType
from ..db import crud
from ..db.database import get_db
from ..services.llm_service import LLMService
from ..services.vector_store import VectorStoreService
import uuid

# Initialize services as singletons
vector_store_service = VectorStoreService()
llm_service = LLMService(vector_store_service)

router = APIRouter()

@router.post("/generate_post", response_model=PostResponse)
def generate_post(request: PostRequest, db: Session = Depends(get_db)):
    # Get or create user to determine user type
    user = crud.get_or_create_user(db, request.user_id)
    
    # Convert database user type to schema UserType enum
    try:
        # Handle the case where user.user_type might be a SQLAlchemy Column object
        if hasattr(user, 'user_type') and user.user_type is not None:
            if isinstance(user.user_type, str):
                user_type = UserType(user.user_type)
            else:
                # If it's a Column or other object, try to get its string value
                user_type = UserType(str(user.user_type.value))
        else:
            user_type = UserType.BEGINNER
    except (ValueError, AttributeError):
        # Default to BEGINNER if conversion fails
        user_type = UserType.BEGINNER
    
    # For PRO users and copywriters, generate multiple options
    if user_type in [UserType.PRO, UserType.COPYWRITER]:
        num_posts = 2
        is_pro = True
    else:
        num_posts = 1
        is_pro = False
    
    # Generate the posts
    generated_posts = []
    similar_posts_text = []
    
    # Get similar posts for context
    similar_docs = vector_store_service.search_similar_posts(request.query, k=3)
    if similar_docs:
        similar_posts_text = [doc.page_content for doc in similar_docs]
    
    # Generate multiple posts if needed
    for _ in range(num_posts):
        post_content = llm_service.generate_post(
            query=request.query,
            client_id=request.client_id or request.user_id,  # Use client_id if available, otherwise user_id
            is_pro_user=is_pro
        )
        
        # Save the post in the database
        post_id = crud.save_post(
            db=db,
            user_id=request.user_id,
            query=request.query,
            content=post_content,
            client_id=request.client_id
        )
        
        # Add to response list
        generated_posts.append(
            GeneratedPost(
                post_id=post_id,
                content=post_content
            )
        )
    
    return PostResponse(
        posts=generated_posts,
        user_type=user_type,
        similar_posts=similar_posts_text if similar_posts_text else None
    )

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

@router.post("/clients", response_model=ClientResponse)
def create_client(user_id: str, name: str, industry: str = None, db: Session = Depends(get_db)):
    # Create a new client for a user (especially useful for copywriters)
    # Type hint fix: industry can be None or str
    industry_val: str | None = industry
    client = crud.create_client(db, user_id, name, industry_val)
    clients = crud.get_clients(db, user_id)
    return ClientResponse(clients=clients)