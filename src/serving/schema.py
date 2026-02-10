"""Request/response schemas for propensity API."""
from typing import List, Optional

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Customer attributes for propensity prediction (UCI Bank style)."""
    age: int = Field(..., ge=0, le=120)
    job: str = "unknown"
    marital: str = "unknown"
    education: str = "unknown"
    default: str = "no"
    balance: int = 0
    housing: str = "no"
    loan: str = "no"
    contact: str = "unknown"
    day: int = Field(..., ge=1, le=31)
    month: str = "may"
    duration: int = Field(..., ge=0)
    campaign: int = Field(..., ge=0)
    pdays: int = Field(..., ge=-2)  # -1 = no previous contact
    previous: int = Field(..., ge=0)
    poutcome: str = "unknown"


class PropensityResponse(BaseModel):
    """Single prediction response."""
    propensity: float = Field(..., ge=0, le=1)
    offer_rankings: Optional[List[dict]] = None  # [{ "offer_id": str, "score": float }, ...]


class BatchPropensityResponse(BaseModel):
    """Batch prediction response."""
    propensities: List[float]
    offer_rankings: Optional[List[List[dict]]] = None
