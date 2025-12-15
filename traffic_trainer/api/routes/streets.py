"""Street information endpoints."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from traffic_trainer.api.models import StreetInfo, StreetListResponse
from traffic_trainer.api.services import model_service

router = APIRouter(prefix="/streets", tags=["Streets"])


@router.get("", response_model=StreetListResponse)
async def list_streets(
    search: Optional[str] = Query(None, description="Search by street name"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List all available streets."""
    if not model_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded")

    streets = model_service.get_all_streets()

    if search:
        search_lower = search.lower()
        streets = [s for s in streets if search_lower in s["street_name"].lower()]

    total = len(streets)
    streets = streets[offset : offset + limit]

    return StreetListResponse(
        total=total,
        streets=[StreetInfo(**s) for s in streets],
    )


@router.get("/{segment_id}", response_model=StreetInfo)
async def get_street(segment_id: int):
    """Get street info by segment ID."""
    if not model_service.is_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded")

    if segment_id not in model_service.street_info:
        raise HTTPException(status_code=404, detail="Segment not found")

    info = model_service.street_info[segment_id]
    return StreetInfo(
        segment_id=segment_id,
        street_id=info.get("street_id", segment_id),
        street_name=info.get("street_name", "Unknown"),
        street_type=info.get("street_type", "Unknown"),
        street_level=info.get("street_level"),
    )
