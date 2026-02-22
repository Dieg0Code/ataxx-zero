from __future__ import annotations

from uuid import UUID

from sqlalchemy.exc import IntegrityError

from api.db.models import ModelVersion
from api.modules.model_versions.repository import ModelVersionRepository
from api.modules.model_versions.schemas import ModelVersionCreateRequest


class ModelVersionService:
    def __init__(self, repository: ModelVersionRepository) -> None:
        self.repository = repository

    async def create_model_version(
        self,
        payload: ModelVersionCreateRequest,
    ) -> ModelVersion:
        version = ModelVersion(**payload.model_dump())
        try:
            created = await self.repository.create(version)
        except IntegrityError as exc:
            raise ValueError("Model version name already exists.") from exc

        if created.is_active:
            await self.repository.deactivate_all_except(keep_id=created.id)
            created = await self.repository.get_by_id(created.id) or created
        return created

    async def get_model_version(self, version_id: UUID) -> ModelVersion | None:
        return await self.repository.get_by_id(version_id)

    async def list_model_versions(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[int, list[ModelVersion]]:
        safe_limit = max(1, min(limit, 200))
        safe_offset = max(0, offset)
        total = await self.repository.count_versions()
        versions = await self.repository.list_recent(limit=safe_limit, offset=safe_offset)
        return total, versions

    async def get_active_model_version(self) -> ModelVersion | None:
        return await self.repository.get_active()

    async def activate_model_version(self, version_id: UUID) -> ModelVersion:
        target = await self.repository.get_by_id(version_id)
        if target is None:
            raise LookupError(f"Model version not found: {version_id}")
        if not target.is_active:
            target.is_active = True
            target = await self.repository.save(target)
        await self.repository.deactivate_all_except(keep_id=target.id)
        return (await self.repository.get_by_id(target.id)) or target
