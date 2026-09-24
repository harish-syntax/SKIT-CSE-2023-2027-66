"""
ORM models.

User Story (Row 2): Build Complaint API -> "Tables for Complaint & Prediction"

Fields mirror the dataset schema already used in data_preprocessing.py
(jaipur_cybercrime_5000_detailed.csv):
Incident_ID, Victim_District, Fraud_Type, Amount_INR, Time_of_Complaint,
Time_to_Withdraw, Target_ATM_Zone, Specific_ATM_Location.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import relationship

from app.database import Base


class Complaint(Base):
    """A single cybercrime complaint record."""

    __tablename__ = "complaints"

    id = Column(Integer, primary_key=True, index=True)
    incident_id = Column(String(50), unique=True, index=True, nullable=False)
    victim_district = Column(String(100), nullable=False)
    fraud_type = Column(String(100), nullable=False)
    amount_inr = Column(Float, nullable=False)
    time_of_complaint = Column(String(10), nullable=False)  # "HH:MM"
    time_to_withdraw = Column(String(20), nullable=True)
    target_atm_zone = Column(String(100), nullable=True)
    specific_atm_location = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    predictions = relationship(
        "Prediction", back_populates="complaint", cascade="all, delete-orphan"
    )


class Prediction(Base):
    """
    Model output for a complaint: the predicted likely cash-withdrawal
    location, produced by the ML pipeline (random_forest_model.py /
    the XGBoost model referenced later in the sprint plan).
    Table is created now; the prediction-generating endpoints are built
    in the later "Build Prediction API" user story.
    """

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    complaint_id = Column(Integer, ForeignKey("complaints.id"), nullable=False)
    predicted_zone = Column(String(100), nullable=True)
    predicted_location = Column(String(200), nullable=True)
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    complaint = relationship("Complaint", back_populates="predictions")
