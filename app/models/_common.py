"""Shared validation rules for the public data contract."""

from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


def safe_relative_path(value: str) -> str:
    """Return a canonical POSIX path which can only point inside the data root."""
    if not value or value != value.strip():
        raise ValueError("路径必须为非空的相对路径")
    if "\\" in value or ":" in value or any(ord(char) < 32 for char in value):
        raise ValueError("路径必须为安全的 POSIX 相对路径")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("路径不得包含绝对路径或上级目录")
    if any(part.endswith((".", " ")) for part in path.parts):
        raise ValueError("路径包含不安全的目录名")
    return path.as_posix()
