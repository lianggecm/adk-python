from sqlalchemy import create_engine, String, Integer, DateTime, PickleType, ForeignKeyConstraint, Engine, MetaData, inspect, func, select, delete
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.orm import Session as DatabaseSessionFactory
from sqlalchemy.exc import ArgumentError, ImportError as SQLAlchemyImportError # Renaming to avoid clash with built-in ImportError
import datetime
from google.genai import types as genai_types
from typing import Any, Optional
import logging
from typing_extensions import override

from .base_artifact_service import BaseArtifactService

logger = logging.getLogger("google_adk." + __name__)

DEFAULT_MAX_KEY_LENGTH = 255

class Base(DeclarativeBase):
  pass

class StorageArtifact(Base):
  __tablename__ = "storage_artifact"

  app_name: Mapped[str] = mapped_column(String(DEFAULT_MAX_KEY_LENGTH), primary_key=True) # type: ignore[var-annotated]
  user_id: Mapped[str] = mapped_column(String(DEFAULT_MAX_KEY_LENGTH), primary_key=True) # type: ignore[var-annotated]
  session_id: Mapped[str] = mapped_column(String(DEFAULT_MAX_KEY_LENGTH), primary_key=True) # type: ignore[var-annotated]
  filename: Mapped[str] = mapped_column(String(DEFAULT_MAX_KEY_LENGTH), primary_key=True) # type: ignore[var-annotated]
  latest_revision_id: Mapped[int] = mapped_column(Integer) # type: ignore[var-annotated]
  create_time: Mapped[datetime.datetime] = mapped_column(DateTime, default=func.now()) # type: ignore[var-annotated]
  update_time: Mapped[datetime.datetime] = mapped_column(DateTime, default=func.now(), onupdate=func.now()) # type: ignore[var-annotated]

  contents: Mapped[list["StorageArtifactContent"]] = relationship("StorageArtifactContent", back_populates="artifact")


class StorageArtifactContent(Base):
  __tablename__ = "storage_artifact_content"

  id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True) # type: ignore[var-annotated]
  artifact_app_name: Mapped[str] = mapped_column(String(DEFAULT_MAX_KEY_LENGTH)) # type: ignore[var-annotated]
  artifact_user_id: Mapped[str] = mapped_column(String(DEFAULT_MAX_KEY_LENGTH)) # type: ignore[var-annotated]
  artifact_session_id: Mapped[str] = mapped_column(String(DEFAULT_MAX_KEY_LENGTH)) # type: ignore[var-annotated]
  artifact_filename: Mapped[str] = mapped_column(String(DEFAULT_MAX_KEY_LENGTH)) # type: ignore[var-annotated]
  revision_id: Mapped[int] = mapped_column(Integer) # type: ignore[var-annotated]
  content: Mapped[genai_types.Part] = mapped_column(PickleType) # type: ignore[var-annotated]
  create_time: Mapped[datetime.datetime] = mapped_column(DateTime, default=func.now()) # type: ignore[var-annotated]

  artifact: Mapped["StorageArtifact"] = relationship("StorageArtifact", back_populates="contents")

  __table_args__ = (
      ForeignKeyConstraint(
          ["artifact_app_name", "artifact_user_id", "artifact_session_id", "artifact_filename"],
          ["storage_artifact.app_name", "storage_artifact.user_id", "storage_artifact.session_id", "storage_artifact.filename"],
          ondelete="CASCADE"
      ),
  )

class DatabaseArtifactService(BaseArtifactService):
  def __init__(self, db_url: str, **kwargs: Any):
    super().__init__()
    try:
      # The return type of create_engine is Engine, but we need to explicitly
      # cast it to Engine because mypy doesn't know that.
      self.db_engine: Engine = create_engine(db_url, **kwargs)  # type: ignore[assignment]
    except (ArgumentError, SQLAlchemyImportError) as e: # Use aliased SQLAlchemyImportError
      # The db_url can contain passwords, so we don't include it in the error.
      logger.error("Failed to create database engine: %s", e)
      raise ValueError(
          "Failed to create database engine. Please check your database URL and"
          " ensure that the correct database drivers are installed."
      ) from e

    self.metadata: MetaData = MetaData()
    self.inspector = inspect(self.db_engine)
    self.database_session_factory: sessionmaker[DatabaseSessionFactory] = sessionmaker(bind=self.db_engine)

    Base.metadata.create_all(self.db_engine)

    @override
    async def save_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        artifact: genai_types.Part,
    ) -> int:
        with self.database_session_factory() as sf: # sf is SessionFactory
            # Check if artifact metadata exists
            stmt = select(StorageArtifact).where(
                StorageArtifact.app_name == app_name,
                StorageArtifact.user_id == user_id,
                StorageArtifact.session_id == session_id,
                StorageArtifact.filename == filename,
            )
            existing_artifact_metadata = sf.execute(stmt).scalar_one_or_none()

            new_revision_id: int
            if existing_artifact_metadata:
                new_revision_id = existing_artifact_metadata.latest_revision_id + 1
                existing_artifact_metadata.latest_revision_id = new_revision_id
                existing_artifact_metadata.update_time = func.now() # type: ignore
                sf.add(existing_artifact_metadata)
            else:
                new_revision_id = 0
                new_artifact_metadata = StorageArtifact(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                    filename=filename,
                    latest_revision_id=new_revision_id,
                    # create_time and update_time will use default func.now()
                )
                sf.add(new_artifact_metadata)

            # Create new artifact content entry
            new_artifact_content = StorageArtifactContent(
                artifact_app_name=app_name,
                artifact_user_id=user_id,
                artifact_session_id=session_id,
                artifact_filename=filename,
                revision_id=new_revision_id,
                content=artifact,
                # create_time will use default func.now()
            )
            sf.add(new_artifact_content)

            sf.commit()
            return new_revision_id

    @override
    async def load_artifact(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        filename: str,
        version: Optional[int] = None,
    ) -> Optional[genai_types.Part]:
        with self.database_session_factory() as sf:
            revision_to_load: Optional[int] = version

            if revision_to_load is None:
                # Get the latest revision_id from StorageArtifact
                artifact_metadata_stmt = select(StorageArtifact.latest_revision_id).where(
                    StorageArtifact.app_name == app_name,
                    StorageArtifact.user_id == user_id,
                    StorageArtifact.session_id == session_id,
                    StorageArtifact.filename == filename,
                )
                latest_revision_id_result = sf.execute(artifact_metadata_stmt).scalar_one_or_none()
                if latest_revision_id_result is None:
                    return None # Artifact metadata not found
                revision_to_load = latest_revision_id_result

            # Load the specific artifact content
            artifact_content_stmt = select(StorageArtifactContent.content).where(
                StorageArtifactContent.artifact_app_name == app_name,
                StorageArtifactContent.artifact_user_id == user_id,
                StorageArtifactContent.artifact_session_id == session_id,
                StorageArtifactContent.artifact_filename == filename,
                StorageArtifactContent.revision_id == revision_to_load,
            )
            result = sf.execute(artifact_content_stmt).scalar_one_or_none()

            return result if result else None

    @override
    async def list_artifact_keys(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> list[str]:
        with self.database_session_factory() as sf:
            stmt = (
                select(StorageArtifact.filename)
                .where(
                    StorageArtifact.app_name == app_name,
                    StorageArtifact.user_id == user_id,
                    StorageArtifact.session_id == session_id,
                )
                .distinct()
            )
            results = sf.execute(stmt).scalars().all()
            return list(results)

    @override
    async def delete_artifact(
        self, *, app_name: str, user_id: str, session_id: str, filename: str
    ) -> None:
        with self.database_session_factory() as sf:
            stmt = delete(StorageArtifact).where(
                StorageArtifact.app_name == app_name,
                StorageArtifact.user_id == user_id,
                StorageArtifact.session_id == session_id,
                StorageArtifact.filename == filename,
            )
            sf.execute(stmt)
            sf.commit()

    @override
    async def list_versions(
        self, *, app_name: str, user_id: str, session_id: str, filename: str
    ) -> list[int]:
        with self.database_session_factory() as sf:
            stmt = (
                select(StorageArtifactContent.revision_id)
                .where(
                    StorageArtifactContent.artifact_app_name == app_name,
                    StorageArtifactContent.artifact_user_id == user_id,
                    StorageArtifactContent.artifact_session_id == session_id,
                    StorageArtifactContent.artifact_filename == filename,
                )
                .order_by(StorageArtifactContent.revision_id)
            )
            results = sf.execute(stmt).scalars().all()
            return list(results)
