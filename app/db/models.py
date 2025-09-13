from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, Text, Enum, Boolean
from sqlalchemy.orm import relationship
import datetime
import uuid

from ..models.schemas import UserType
from .database import Base

class User(Base):
    __tablename__ = "users"
    user_id = Column(String, primary_key=True)
    user_type = Column(Enum(UserType), default=UserType.BEGINNER)
    post_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.now)
    clients = relationship("Client", back_populates="user")
    posts = relationship("Post", back_populates="user")

class Client(Base):
    __tablename__ = "clients"
    client_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.user_id"))
    name = Column(String, nullable=False)
    industry = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.now)
    user = relationship("User", back_populates="clients")
    posts = relationship("Post", back_populates="client")

class Post(Base):
    __tablename__ = "posts"
    post_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.user_id"))
    client_id = Column(String, ForeignKey("clients.client_id"), nullable=True)
    query = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    chosen = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.now)
    user = relationship("User", back_populates="posts")
    client = relationship("Client", back_populates="posts")