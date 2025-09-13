from sqlalchemy.orm import Session
from . import models
from ..models.schemas import UserType, Client as ClientSchema
import uuid

def get_user(db: Session, user_id: str):
    return db.query(models.User).filter(models.User.user_id == user_id).first()

def create_user(db: Session, user_id: str, user_type: UserType = UserType.BEGINNER):
    db_user = models.User(user_id=user_id, user_type=user_type)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_or_create_user(db: Session, user_id: str):
    user = get_user(db, user_id)
    if not user:
        user = create_user(db, user_id)
    return user

def get_clients(db: Session, user_id: str):
    db_clients = db.query(models.Client).filter(models.Client.user_id == user_id).all()
    return [ClientSchema(client_id=c.client_id, name=c.name, industry=c.industry, created_at=c.created_at) 
            for c in db_clients]

def create_client(db: Session, user_id: str, name: str, industry=None):
    client_id = str(uuid.uuid4())
    db_client = models.Client(client_id=client_id, user_id=user_id, name=name, industry=industry)
    db.add(db_client)
    db.commit()
    db.refresh(db_client)
    return db_client

def save_post(db: Session, user_id: str, query: str, content: str, client_id=None):
    post_id = str(uuid.uuid4())
    db_post = models.Post(
        post_id=post_id,
        user_id=user_id,
        client_id=client_id,
        query=query,
        content=content
    )
    db.add(db_post)
    user = get_user(db, user_id)
    user.post_count += 1
    db.commit()
    return post_id

def save_post_choice(db: Session, post_id: str):
    db_post = db.query(models.Post).filter(models.Post.post_id == post_id).first()
    if db_post:
        db_post.chosen = True
        db.commit()
        return True
    return False