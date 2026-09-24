"""
CRUD operations for complaints. Kept separate from the router so the
data-access logic is easy to unit test and reuse (e.g. later from the
Prediction API).
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from app import models, schemas


def get_complaint(db: Session, complaint_id: int) -> Optional[models.Complaint]:
    return db.query(models.Complaint).filter(models.Complaint.id == complaint_id).first()


def get_complaint_by_incident_id(db: Session, incident_id: str) -> Optional[models.Complaint]:
    return (
        db.query(models.Complaint)
        .filter(models.Complaint.incident_id == incident_id)
        .first()
    )


def get_complaints(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    victim_district: Optional[str] = None,
    fraud_type: Optional[str] = None,
) -> List[models.Complaint]:
    query = db.query(models.Complaint)
    if victim_district:
        query = query.filter(models.Complaint.victim_district == victim_district)
    if fraud_type:
        query = query.filter(models.Complaint.fraud_type == fraud_type)
    return query.order_by(models.Complaint.id.desc()).offset(skip).limit(limit).all()


def create_complaint(db: Session, complaint: schemas.ComplaintCreate) -> models.Complaint:
    db_complaint = models.Complaint(**complaint.model_dump())
    db.add(db_complaint)
    db.commit()
    db.refresh(db_complaint)
    return db_complaint


def update_complaint(
    db: Session, db_complaint: models.Complaint, updates: schemas.ComplaintUpdate
) -> models.Complaint:
    for field, value in updates.model_dump(exclude_unset=True).items():
        setattr(db_complaint, field, value)
    db.commit()
    db.refresh(db_complaint)
    return db_complaint


def delete_complaint(db: Session, db_complaint: models.Complaint) -> None:
    db.delete(db_complaint)
    db.commit()
