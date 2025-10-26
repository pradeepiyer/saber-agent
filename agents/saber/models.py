"""Request and response models for Saber Agent HTTP endpoints."""

from pydantic import BaseModel, Field


class SaberRequest(BaseModel):
    """Request model for Saber Agent."""

    query: str = Field(..., description="Baseball statistics query")
    session_id: str | None = Field(None, description="Optional session ID for continuation")


class SaberResponse(BaseModel):
    """Response model for Saber Agent."""

    response: str = Field(..., description="Statistics response")
    session_id: str = Field(..., description="Session ID for continuation")
