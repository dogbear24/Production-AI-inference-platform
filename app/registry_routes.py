from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.registry_models import ModelRecord
from app.registry_schemas import (
    ModelCreate,
    ModelResponse,
)


router = APIRouter(
    prefix="/models",
    tags=["model registry"],
)


@router.post(
    "",
    response_model=ModelResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_model(
    model: ModelCreate,
    db: Session = Depends(get_db),
):

    existing = db.scalar(
        select(ModelRecord).where(
            ModelRecord.name == model.name,
            ModelRecord.version == model.version,
        )
    )

    if existing:
        raise HTTPException(
            status_code=409,
            detail=(
                "This model name/version "
                "already exists."
            ),
        )

    record = ModelRecord(
        name=model.name,
        version=model.version,
        framework=model.framework,
        task=model.task,
        artifact_path=model.artifact_path,
    )

    db.add(record)

    db.commit()

    db.refresh(record)

    return record


@router.get(
    "",
    response_model=list[ModelResponse],
)
def list_models(
    db: Session = Depends(get_db),
):

    result = db.scalars(
        select(ModelRecord)
        .order_by(ModelRecord.created_at.desc())
    )

    return list(result)


@router.get(
    "/active/{model_name}",
    response_model=ModelResponse,
)
def get_active_model(
    model_name: str,
    db: Session = Depends(get_db),
):

    record = db.scalar(
        select(ModelRecord).where(
            ModelRecord.name == model_name,
            ModelRecord.is_active.is_(True),
        )
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="No active version found.",
        )

    return record


@router.get(
    "/{model_id}",
    response_model=ModelResponse,
)
def get_model(
    model_id: int,
    db: Session = Depends(get_db),
):

    record = db.get(
        ModelRecord,
        model_id,
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Model not found.",
        )

    return record


@router.post(
    "/{model_id}/activate",
    response_model=ModelResponse,
)
def activate_model(
    model_id: int,
    db: Session = Depends(get_db),
):

    record = db.get(
        ModelRecord,
        model_id,
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Model not found.",
        )

    other_versions = db.scalars(
        select(ModelRecord).where(
            ModelRecord.name == record.name,
            ModelRecord.id != record.id,
        )
    )

    for other in other_versions:
        other.is_active = False

    record.is_active = True

    db.commit()

    db.refresh(record)

    return record