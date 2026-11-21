from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()

try:
    from pgvector.sqlalchemy import Vector
    VectorType = Vector(384)
except Exception:
    VectorType = Text


class DocumentRecord(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChunkRecord(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(VectorType, nullable=True)
    metadata_json = Column(Text, default="{}")


class ProcessModelRecord(Base):
    __tablename__ = "process_models"

    id = Column(String, primary_key=True)
    document_id = Column(String, nullable=True, index=True)
    title = Column(String, nullable=False)
    version = Column(String, default="1.0")
    model_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
