from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ModelRecord(Base):

    __tablename__ = "model_registry"

    __table_args__ = (
        UniqueConstraint(
            "name",
            "version",
            name="uq_model_name_version",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        index=True,
    )

    version: Mapped[str] = mapped_column(
        String(50),
    )

    framework: Mapped[str] = mapped_column(
        String(50),
    )

    task: Mapped[str] = mapped_column(
        String(100),
    )

    artifact_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
    )