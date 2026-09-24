"""
FastAPI entrypoint.

User Story (Row 1): Setup project and database
- Creates the FastAPI app
- Creates all tables (Complaint, Prediction) on startup

User Story (Row 2): Build Complaint API
- Mounts the /complaints router
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import Base, engine
from app.routers import complaints

# Import models so they register on Base.metadata before create_all() runs
from app import models  # noqa: F401

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Cybercrime Complaint & Cash-Withdrawal Prediction API",
    description=(
        "Backend for the Predictive Analytics Framework for Cybercrime "
        "Complaints — forecasts likely cash-withdrawal locations to "
        "enable timely, proactive intervention."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(complaints.router)


@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "cybercrime-complaint-api"}
