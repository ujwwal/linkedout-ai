# setup_db.py
from app.db.database import engine
from app.db import models

def setup_database():
    models.Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

if __name__ == "__main__":
    setup_database()