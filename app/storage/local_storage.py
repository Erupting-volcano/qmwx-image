"""Stage a complete object and atomically publish its directory on the same disk."""

from contextlib import AbstractContextManager
import json
import logging
import os
from pathlib import Path
import shutil
from uuid import UUID, uuid4

from app.models.object_profile import ObjectProfile


logger = logging.getLogger(__name__)


class ProfileNotFoundError(Exception):
    pass


class CorruptProfileError(Exception):
    pass


class ObjectTransaction(AbstractContextManager):
    def __init__(self, storage: "LocalStorage", object_id: UUID):
        self.storage = storage
        self.object_id = object_id
        self.staging_base = storage.data_dir / ".staging"
        self.data_dir = self.staging_base / str(uuid4())
        self.object_dir = self.data_dir / "objects" / str(object_id)
        self.images_dir = self.object_dir / "images"
        self.committed = False

    def __enter__(self) -> "ObjectTransaction":
        self.images_dir.mkdir(parents=True, exist_ok=False)
        return self

    def save_profile(self, profile: ObjectProfile) -> dict:
        if profile.object_id != self.object_id:
            raise ValueError("Transaction object ID mismatch")
        # Validate the final serialized form, including all nested module output.
        payload = profile.model_dump(mode="json")
        ObjectProfile.model_validate(payload)
        content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        temporary_path = self.object_dir / f"profile.{uuid4()}.tmp"
        with temporary_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, self.object_dir / "profile.json")
        return json.loads(content)

    def commit(self) -> None:
        destination = self.storage.object_dir(self.object_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError("Object ID already exists")
        # Both directories are inside data_dir: no cross-volume copy or half-published profile.
        os.replace(self.object_dir, destination)
        self.committed = True

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        # Resolve and check the exact recursive deletion target before removal.
        target = self.data_dir.resolve()
        allowed = self.staging_base.resolve()
        if target != allowed and target.is_relative_to(allowed) and target.exists():
            try:
                shutil.rmtree(target)
            except OSError:
                # A published profile must still be returned if only staging cleanup fails.
                logger.exception("Staging cleanup failed for transaction %s", self.data_dir.name)


class LocalStorage:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir).resolve()

    def object_dir(self, object_id: UUID) -> Path:
        safe_id = UUID(str(object_id))
        return self.data_dir / "objects" / str(safe_id)

    def transaction(self, object_id: UUID) -> ObjectTransaction:
        return ObjectTransaction(self, object_id)

    def read_profile(self, object_id: UUID) -> tuple[ObjectProfile, bytes]:
        profile_path = self.object_dir(object_id) / "profile.json"
        try:
            content = profile_path.read_bytes()
        except FileNotFoundError as exc:
            raise ProfileNotFoundError from exc
        try:
            profile = ObjectProfile.model_validate_json(content)
            if profile.object_id != object_id:
                raise ValueError("Stored object ID mismatch")
        except (ValueError, TypeError) as exc:
            raise CorruptProfileError from exc
        return profile, content
