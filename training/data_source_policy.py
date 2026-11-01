"""Guards preventing training from using sources without verified reuse rights."""

from __future__ import annotations

from pathlib import Path


def reject_restricted_dataset(path: Path) -> None:
    """Fail explicitly when a removed or unlicensed dataset is requested."""
    normalized = path.resolve()
    parts = {part.casefold() for part in normalized.parts}
    if normalized.name.casefold() == "used car dataset.csv".casefold():
        raise PermissionError(
            "Dataset A is removed because its reuse licence is unverified. "
            "Use the approved Dataset B source or obtain permission first."
        )
    if "dataset_c_github" in parts or normalized.name.casefold() == "used_cars.csv":
        raise PermissionError(
            "Dataset C is excluded because its reuse licence is unverified. "
            "Obtain permission before using it for training."
        )
