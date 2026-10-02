from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelCreate(BaseModel):

    name: str = Field(
        min_length=1,
        max_length=100,
    )

    version: str = Field(
        min_length=1,
        max_length=50,
    )

    framework: str = "pytorch"

    task: str = "image-classification"

    artifact_path: str | None = None


class ModelResponse(ModelCreate):

    id: int

    is_active: bool

    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )