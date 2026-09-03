"""Exact observation identity and cache support for resumable evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _canonical_float(value: float) -> str:
    return format(float(value), ".17g")


@dataclass(frozen=True)
class ObservationKey:
    """Immutable identity for one benchmark observation."""

    model_id: str
    z: int
    gravity: str
    tier: int
    seed: int
    grid_key: str
    experiment_hash: str

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ObservationKey":
        required = ("model_id", "z", "gravity", "tier", "seed", "grid_key", "experiment_hash")
        missing = [name for name in required if config.get(name) is None]
        if missing:
            raise ValueError(f"Observation identity missing fields: {', '.join(missing)}")
        return cls(
            model_id=str(config["model_id"]),
            z=int(config["z"]),
            gravity=_canonical_float(config["gravity"]),
            tier=int(config["tier"]),
            seed=int(config["seed"]),
            grid_key=str(config["grid_key"]),
            experiment_hash=str(config["experiment_hash"]),
        )


@dataclass
class ObservationRecord:
    """Cached result plus provenance; invalid results are first-class records."""

    key: ObservationKey
    data: Dict[str, Any]
    raw_output: str = ""


class ResumeObservationStore:
    """In-memory exact-key index populated from one or more raw artifacts."""

    def __init__(self, records: Optional[Iterable[ObservationRecord]] = None):
        self._records: Dict[ObservationKey, ObservationRecord] = {}
        for record in records or ():
            self.put(record)

    def get(self, key: ObservationKey) -> Optional[ObservationRecord]:
        return self._records.get(key)

    def put(self, record: ObservationRecord) -> None:
        if record.key in self._records:
            raise ValueError(f"Duplicate observation key: {record.key}")
        self._records[record.key] = record

    def __len__(self) -> int:
        return len(self._records)

    @classmethod
    def from_records(cls, records: Iterable[Dict[str, Any]]) -> "ResumeObservationStore":
        store = cls()
        for record in records:
            key = ObservationKey.from_config(record)
            store.put(ObservationRecord(key=key, data=dict(record),
                                        raw_output=str(record.get("raw_output", "") or "")))
        return store

    @classmethod
    def from_raw_file(cls, path: str, model_id: Optional[str] = None) -> "ResumeObservationStore":
        """Index structured metadata blocks without rescanning on each lookup."""
        text = Path(path).read_text(encoding="utf-8")
        records = []
        for block in text.split("=== RAW STREAM ENTRY")[1:]:
            metadata = {}
            for line in block.splitlines():
                if line.strip().startswith("EnvironmentMetadata:"):
                    for item in line.split(":", 1)[1].split(","):
                        if "=" in item:
                            name, value = item.strip().split("=", 1)
                            metadata[name.strip()] = value.strip()
            if model_id is not None:
                metadata["model_id"] = model_id
            if not all(metadata.get(name) is not None for name in
                       ("model_id", "z", "gravity", "tier", "seed", "grid_key", "experiment_hash")):
                continue
            raw = ""
            if "[MODEL OUTPUT]:" in block:
                raw = block.split("[MODEL OUTPUT]:", 1)[1].split("[GROUND TRUTH EXPECTED]:", 1)[0].strip()
            metadata["raw_output"] = raw
            key = ObservationKey.from_config(metadata)
            store_record = ObservationRecord(key=key, data=metadata, raw_output=raw)
            # Duplicate semantic observations are corruption, not a retry policy.
            records.append(store_record)
        return cls(records)
