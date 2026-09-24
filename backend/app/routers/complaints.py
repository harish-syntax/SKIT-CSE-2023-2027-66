"""
Complaint API.

User Story (Row 2): Build Complaint API
Endpoints:
  POST   /complaints/            -> create a complaint
  GET    /complaints/            -> list complaints (paginated, filterable)
  GET    /complaints/{id}        -> get one complaint
  PATCH  /complaints/{id}        -> update a complaint
  DELETE /complaints/{id}        -> delete a complaint
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app import crud, schemas
from app.database import get_db

router = APIRouter(prefix="/complaints", tags=["Complaints"])


@router.post("/", response_model=schemas.ComplaintOut, status_code=status.HTTP_201_CREATED)
def create_complaint(complaint: schemas.ComplaintCreate, db: Session = Depends(get_db)):
    existing = crud.get_complaint_by_incident_id(db, complaint.incident_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Complaint with incident_id '{complaint.incident_id}' already exists.",
        )
    return crud.create_complaint(db, complaint)


@router.get("/", response_model=List[schemas.ComplaintOut])
def list_complaints(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    victim_district: Optional[str] = None,
    fraud_type: Optional[str] = None,
    db: Session = Depends(get_db),
):
    return crud.get_complaints(
        db, skip=skip, limit=limit, victim_district=victim_district, fraud_type=fraud_type
    )


@router.get("/{complaint_id}", response_model=schemas.ComplaintOut)
def get_complaint(complaint_id: int, db: Session = Depends(get_db)):
    db_complaint = crud.get_complaint(db, complaint_id)
    if not db_complaint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complaint not found")
    return db_complaint


@router.patch("/{complaint_id}", response_model=schemas.ComplaintOut)
def update_complaint(
    complaint_id: int, updates: schemas.ComplaintUpdate, db: Session = Depends(get_db)
):
    db_complaint = crud.get_complaint(db, complaint_id)
    if not db_complaint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complaint not found")
    return crud.update_complaint(db, db_complaint, updates)


@router.delete("/{complaint_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_complaint(complaint_id: int, db: Session = Depends(get_db)):
    db_complaint = crud.get_complaint(db, complaint_id)
    if not db_complaint:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Complaint not found")
    crud.delete_complaint(db, db_complaint)
    return None
