"""
Checkpoint management for ONFRA PFAS pipeline.

Provides run folder creation and Parquet/JSON checkpoint save/load.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .errors import CheckpointError
from .logging import setup_run_logger, log_config_snapshot
from .utils import generate_run_id, file_hash, ensure_dir


@dataclass
class RunMeta:
    """Metadata for a run."""

    run_id: str
    created_at: str
    input_files: list[dict]  # [{path, hash, size_mb, scan_count}, ...]
    config_snapshot: dict
    gpu_mode: str
    backend: str  # "numpy" or "cupy"
    gpu_info: dict | None
    suspect_list_info: dict | None  # {path, hash, version}
    steps: list[dict] = field(default_factory=list)  # [{name, status, duration_s, ...}]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "input_files": self.input_files,
            "config_snapshot": self.config_snapshot,
            "gpu_mode": self.gpu_mode,
            "backend": self.backend,
            "gpu_info": self.gpu_info,
            "suspect_list_info": self.suspect_list_info,
            "steps": self.steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RunMeta:
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            created_at=data["created_at"],
            input_files=data["input_files"],
            config_snapshot=data["config_snapshot"],
            gpu_mode=data["gpu_mode"],
            backend=data["backend"],
            gpu_info=data.get("gpu_info"),
            suspect_list_info=data.get("suspect_list_info"),
            steps=data.get("steps", []),
        )


@dataclass
class StepMeta:
    """Metadata for a pipeline step."""

    step_name: str
    step_number: int
    status: str  # "completed", "failed", "skipped"
    started_at: str
    completed_at: str | None = None
    duration_seconds: float = 0.0
    input_feature_count: int | None = None
    output_feature_count: int | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "step_name": self.step_name,
            "step_number": self.step_number,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "input_feature_count": self.input_feature_count,
            "output_feature_count": self.output_feature_count,
            **self.extra,
        }


class RunContext:
    """Context for a single pipeline run."""

    # Step names and their checkpoint files
    STEPS = {
        "featurefinding": ("01_featurefinding.parquet", "01_featurefinding_meta.json"),
        "prioritization": ("02_prioritization.parquet", "02_prioritization_meta.json"),
        "visualization": ("03_viz_payload.json", "03_viz_meta.json"),
    }

    def __init__(self, run_dir: Path, run_meta: RunMeta):
        """
        Initialize run context.

        Args:
            run_dir: Root directory for the run
            run_meta: Run metadata
        """
        self.run_dir = run_dir
        self.run_meta = run_meta
        self.logger = setup_run_logger(run_dir)

        # Directory structure
        self.input_dir = run_dir / "input"
        self.checkpoints_dir = run_dir / "checkpoints"
        self.reports_dir = run_dir / "reports"
        self.logs_dir = run_dir / "logs"

    def save_checkpoint(
        self,
        step: str,
        data: pd.DataFrame | dict,
        meta: StepMeta,
    ) -> Path:
        """
        Save checkpoint for a step.

        Args:
            step: Step name (featurefinding, prioritization, visualization)
            data: DataFrame or dict to save
            meta: Step metadata

        Returns:
            Path to checkpoint file
        """
        if step not in self.STEPS:
            raise CheckpointError("save", step, f"Unknown step: {step}")

        data_file, meta_file = self.STEPS[step]
        data_path = self.checkpoints_dir / data_file
        meta_path = self.checkpoints_dir / meta_file

        try:
            # Save data
            if isinstance(data, pd.DataFrame):
                data.to_parquet(data_path, index=False)
            else:
                with open(data_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            # Save metadata
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta.to_dict(), f, indent=2, ensure_ascii=False)

            # Update run meta
            self.run_meta.steps.append(meta.to_dict())
            self._save_run_meta()

            self.logger.info(f"Checkpoint saved: {step}")
            return data_path

        except Exception as e:
            raise CheckpointError("save", str(data_path), str(e)) from e

    def load_checkpoint(
        self,
        step: str,
    ) -> tuple[pd.DataFrame | dict, StepMeta]:
        """
        Load checkpoint for a step.

        Args:
            step: Step name

        Returns:
            Tuple of (data, metadata)
        """
        if step not in self.STEPS:
            raise CheckpointError("load", step, f"Unknown step: {step}")

        data_file, meta_file = self.STEPS[step]
        data_path = self.checkpoints_dir / data_file
        meta_path = self.checkpoints_dir / meta_file

        if not data_path.exists():
            raise CheckpointError("load", str(data_path), "File not found")

        try:
            # Load data
            if data_path.suffix == ".parquet":
                data = pd.read_parquet(data_path)
            else:
                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

            # Load metadata
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_dict = json.load(f)

            meta = StepMeta(
                step_name=meta_dict["step_name"],
                step_number=meta_dict["step_number"],
                status=meta_dict["status"],
                started_at=meta_dict["started_at"],
                completed_at=meta_dict.get("completed_at"),
                duration_seconds=meta_dict.get("duration_seconds", 0),
                input_feature_count=meta_dict.get("input_feature_count"),
                output_feature_count=meta_dict.get("output_feature_count"),
            )

            return data, meta

        except Exception as e:
            raise CheckpointError("load", str(data_path), str(e)) from e

    def has_checkpoint(self, step: str) -> bool:
        """Check if checkpoint exists for a step."""
        if step not in self.STEPS:
            return False
        data_file, _ = self.STEPS[step]
        return (self.checkpoints_dir / data_file).exists()

    def save_report(self, filename: str, content: str) -> Path:
        """
        Save a report file.

        Args:
            filename: Report filename
            content: Report content (HTML, etc.)

        Returns:
            Path to report file
        """
        report_path = self.reports_dir / filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        return report_path

    def _save_run_meta(self) -> None:
        """Save run metadata to file."""
        meta_path = self.run_dir / "run_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.run_meta.to_dict(), f, indent=2, ensure_ascii=False)


class RunManager:
    """Manager for creating and loading runs."""

    def __init__(self, base_dir: Path | str = "runs"):
        """
        Initialize run manager.

        Args:
            base_dir: Base directory for all runs
        """
        self.base_dir = Path(base_dir)

    def create_run(
        self,
        config: dict,
        input_files: list[Path],
        gpu_mode: str,
        backend: str,
        gpu_info: dict | None = None,
        suspect_list_path: Path | None = None,
        run_id: str | None = None,
    ) -> RunContext:
        """
        Create a new run.

        Args:
            config: Pipeline configuration dict
            input_files: List of input mzML file paths
            gpu_mode: GPU mode string
            backend: Backend name (numpy/cupy)
            gpu_info: GPU information dict
            suspect_list_path: Path to suspect list file
            run_id: Optional custom run ID

        Returns:
            RunContext for the new run
        """
        # Generate run ID
        if run_id is None:
            run_id = generate_run_id()

        # Create run directory structure
        run_dir = self.base_dir / run_id
        ensure_dir(run_dir / "input")
        ensure_dir(run_dir / "checkpoints")
        ensure_dir(run_dir / "reports")
        ensure_dir(run_dir / "logs")

        # Gather input file info
        input_file_info = []
        for path in input_files:
            path = Path(path)
            info = {
                "path": str(path.absolute()),
                "filename": path.name,
                "hash": file_hash(path),
                "size_mb": path.stat().st_size / (1024 * 1024),
            }
            input_file_info.append(info)

            # Create symlink or copy reference in input dir
            link_path = run_dir / "input" / path.name
            if not link_path.exists():
                try:
                    link_path.symlink_to(path.absolute())
                except OSError:
                    # Symlinks may not work on some systems, just store path
                    with open(link_path.with_suffix(".path.txt"), "w") as f:
                        f.write(str(path.absolute()))

        # Gather suspect list info
        suspect_info = None
        if suspect_list_path:
            suspect_list_path = Path(suspect_list_path)
            if suspect_list_path.exists():
                suspect_info = {
                    "path": str(suspect_list_path.absolute()),
                    "hash": file_hash(suspect_list_path),
                }

        # Create run metadata
        run_meta = RunMeta(
            run_id=run_id,
            created_at=datetime.now().isoformat(),
            input_files=input_file_info,
            config_snapshot=config,
            gpu_mode=gpu_mode,
            backend=backend,
            gpu_info=gpu_info,
            suspect_list_info=suspect_info,
        )

        # Create context
        context = RunContext(run_dir, run_meta)

        # Save initial metadata
        context._save_run_meta()

        # Save config snapshot
        log_config_snapshot(
            context.logger,
            config,
            run_dir / "logs" / "config_snapshot.json",
        )

        context.logger.info(f"Run created: {run_id}")
        return context

    def load_run(self, run_id: str) -> RunContext:
        """
        Load an existing run.

        Args:
            run_id: Run ID to load

        Returns:
            RunContext for the run
        """
        run_dir = self.base_dir / run_id

        if not run_dir.exists():
            raise CheckpointError("load", str(run_dir), "Run not found")

        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            raise CheckpointError("load", str(meta_path), "Run metadata not found")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        run_meta = RunMeta.from_dict(meta_dict)
        return RunContext(run_dir, run_meta)

    def list_runs(self) -> list[dict]:
        """
        List all runs.

        Returns:
            List of run info dicts
        """
        runs = []
        if not self.base_dir.exists():
            return runs

        for run_dir in self.base_dir.iterdir():
            if not run_dir.is_dir():
                continue

            meta_path = run_dir / "run_meta.json"
            if not meta_path.exists():
                continue

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                runs.append({
                    "run_id": meta["run_id"],
                    "created_at": meta["created_at"],
                    "input_count": len(meta["input_files"]),
                    "steps_completed": len(meta.get("steps", [])),
                })
            except Exception:
                continue

        # Sort by creation time (newest first)
        runs.sort(key=lambda x: x["created_at"], reverse=True)
        return runs

    def delete_run(self, run_id: str) -> None:
        """
        Delete a run.

        Args:
            run_id: Run ID to delete
        """
        run_dir = self.base_dir / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
