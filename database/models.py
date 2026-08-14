from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase): pass


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(50), default="api")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    review_id: Mapped[int] = mapped_column(ForeignKey("reviews.id"))
    sentiment: Mapped[str] = mapped_column(String(20)); confidence: Mapped[float] = mapped_column(Float)
    topic: Mapped[str] = mapped_column(String(40)); risk_level: Mapped[str] = mapped_column(String(20)); risk_score: Mapped[int] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(String(40)); created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True); model_name: Mapped[str] = mapped_column(String(80)); version: Mapped[str] = mapped_column(String(40)); macro_f1: Mapped[float | None] = mapped_column(Float, nullable=True); created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class BatchJob(Base):
    __tablename__ = "batch_jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True); row_count: Mapped[int] = mapped_column(Integer); status: Mapped[str] = mapped_column(String(20)); created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

