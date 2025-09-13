from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime

class UserType(str, Enum):
    COPYWRITER = "copywriter"
    NORMAL = "normal"
    PRO = "pro"
    BEGINNER = "beginner"

class Client(BaseModel):
    client_id: str
    name: str
    industry: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

class PostRequest(BaseModel):
    user_id: str
    query: str
    client_id: Optional[str] = None

class PostChoice(BaseModel):
    user_id: str
    post_id: str
    chosen_index: int
    client_id: Optional[str] = None

class GeneratedPost(BaseModel):
    post_id: str
    content: str
    score: Optional[float] = None

class PostResponse(BaseModel):
    posts: List[GeneratedPost]
    user_type: UserType
    similar_posts: Optional[List[str]] = None

class ClientResponse(BaseModel):
    clients: List[Client]