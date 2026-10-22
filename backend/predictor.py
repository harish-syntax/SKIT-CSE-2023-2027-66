"""
Prediction logic.

Baseline: most frequent (zone, location) among past complaints with the same
fraud_type (and same district when enough data exists). Confidence = that
pair's share of the matching complaints.

To plug in the trained ML model later, replace the body of `predict()` and
keep the return shape: (zone, location, confidence).
"""

from collections import Counter
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from app import models

Result = Tuple[Optional[str], Optional[str], Optional[float]]


def predict(db: Session, complaint: models.Complaint) -> Result:
    base = (
        db.query(models.Complaint)
        .filter(
            models.Complaint.id != complaint.id,
            models.Complaint.fraud_type == complaint.fraud_type,
            models.Complaint.target_atm_zone.isnot(None),
        )
    )
    same_district = base.filter(
        models.Complaint.victim_district == complaint.victim_district
    ).all()
    rows = same_district if len(same_district) >= 5 else base.all()
    if not rows:
        return None, None, None

    counts = Counter((r.target_atm_zone, r.specific_atm_location) for r in rows)
    (zone, location), n = counts.most_common(1)[0]
    return zone, location, round(n / len(rows), 4)
