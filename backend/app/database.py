"""
Database setup for the Cybercrime Complaint Prediction backend.

User Story (Row 1): Setup project and database
- Initializes the SQLAlchemy engine, session factory and declarative Base
  used by every model in the app.

Default: SQLite file `cybercrime.db` so the project runs with zero external
setup. To use Postgres/MySQL later, just change DATABASE_URL (e.g. via an
environment variable) — nothing else in the app needs to change.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cybercrime.db")

# check_same_thread is only needed for SQLite
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
