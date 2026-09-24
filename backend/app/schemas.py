"""
Pydantic schemas used by the Complaint API for request validation and
response serialization.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class ComplaintBase(BaseModel):
    incident_id: str = Field(..., max_length=50, examples=["INC-000123"])
    victim_district: str = Field(..., max_length=100, examples=["Jaipur"])
    fraud_type: str = Field(..., max_length=100, examples=["UPI Fraud"])
    amount_inr: float = Field(..., gt=0, examples=[15000.0])
    time_of_complaint: str = Field(..., max_length=10, examples=["14:30"])
    time_to_withdraw: Optional[str] = Field(None, max_length=20, examples=["02:15"])
    target_atm_zone: Optional[str] = Field(None, max_length=100)
    specific_atm_location: Optional[str] = Field(None, max_length=200)


class ComplaintCreate(ComplaintBase):
    """Payload for POST /complaints"""
    pass


class ComplaintUpdate(BaseModel):
    """Payload for PATCH /complaints/{id} — every field optional."""
    victim_district: Optional[str] = None
    fraud_type: Optional[str] = None
    amount_inr: Optional[float] = Field(None, gt=0)
    time_of_complaint: Optional[str] = None
    time_to_withdraw: Optional[str] = None
    target_atm_zone: Optional[str] = None
    specific_atm_location: Optional[str] = None


class ComplaintOut(ComplaintBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
