"""Provider-agnostic run observability adapters.

This module is intentionally dependency-light: if W&B is not installed or
not enabled, all logging calls become no-ops.
"""

from __future__ import annotations

import os
import time
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


class BaseObservabilityBackend:
    """Provider-agnostic interface for run lifecycle hooks."""

    def start_run(self, run_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        return None

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        return None

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        return None

    def finish_run(self, status: str = "completed") -> None:
        return None


class NoOpObservabilityBackend(BaseObservabilityBackend):
    """No-op backend for local runs and tests."""


class WandBObservabilityBackend(BaseObservabilityBackend):
    """Optional Weights & Biases backend."""

    def __init__(self, project: str, entity: Optional[str] = None, mode: str = "online"):
        self._wandb = None
        self._run = None
        self.project = project
        self.entity = entity
        self.mode = mode

    def _import_wandb(self):
        if self._wandb is None:
            import wandb

            self._wandb = wandb
        return self._wandb

    def start_run(self, run_name: str, config: Optional[Dict[str, Any]] = None) -> None:
        wandb = self._import_wandb()
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=run_name,
            config=config or {},
            mode=self.mode,
            reinit=True,
        )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._run is None:
            return
        self._wandb.log(metrics, step=step)

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        if self._run is None:
            return
        artifact = self._wandb.Artifact(name or os.path.basename(path), type="artifact")
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    def finish_run(self, status: str = "completed") -> None:
        if self._run is not None:
            self._wandb.finish(exit_code=0 if status == "completed" else 1)
            self._run = None


@dataclass
class ObservabilityConfig:
    backend: str = "none"
    run_name: str = "better-ai-train"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"


class ObservabilityAdapter:
    """Fault-tolerant wrapper around a concrete observability backend."""

    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.enabled = False
        self.backend: BaseObservabilityBackend = NoOpObservabilityBackend()

        backend = (config.backend or "none").lower()
        if backend == "wandb":
            try:
                project = config.wandb_project or os.getenv("WANDB_PROJECT")
                if not project:
                    return
                entity = config.wandb_entity or os.getenv("WANDB_ENTITY")
                mode = config.wandb_mode or os.getenv("WANDB_MODE", "online")
                self.backend = WandBObservabilityBackend(project=project, entity=entity, mode=mode)
                self.enabled = True
            except Exception:
                self.backend = NoOpObservabilityBackend()
                self.enabled = False

    @classmethod
    def from_config(cls, config_obj: Any) -> "ObservabilityAdapter":
        backend = getattr(config_obj, "observability_backend", os.getenv("BETTER_AI_OBSERVABILITY_BACKEND", "none"))
        run_name = getattr(config_obj, "run_name", os.getenv("BETTER_AI_RUN_NAME", "better-ai-train"))
        wandb_project = getattr(config_obj, "wandb_project", os.getenv("WANDB_PROJECT"))
        wandb_entity = getattr(config_obj, "wandb_entity", os.getenv("WANDB_ENTITY"))
        wandb_mode = getattr(config_obj, "wandb_mode", os.getenv("WANDB_MODE", "online"))
        return cls(
            ObservabilityConfig(
                backend=backend,
                run_name=run_name,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                wandb_mode=wandb_mode,
            )
        )

    def start_run(self, config: Optional[Dict[str, Any]] = None) -> None:
        try:
            self.backend.start_run(self.config.run_name, config=config)
        except Exception:
            pass

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        try:
            self.backend.log_metrics(metrics, step=step)
        except Exception:
            pass

    def log_artifact(self, path: str, name: Optional[str] = None) -> None:
        try:
            self.backend.log_artifact(path=path, name=name)
        except Exception:
            pass

    def finish_run(self, status: str = "completed") -> None:
        try:
            self.backend.finish_run(status=status)
        except Exception:
            pass


def collect_gpu_stats() -> Dict[str, float]:
    """Collect GPU stats with graceful fallback if telemetry is unavailable."""
    stats: Dict[str, float] = {
        "gpu/memory_allocated_gb": 0.0,
        "gpu/memory_reserved_gb": 0.0,
        "gpu/utilization_pct": 0.0,
    }

    if not torch.cuda.is_available():
        return stats

    try:
        stats["gpu/memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
        stats["gpu/memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)
    except Exception:
        pass

    # Best-effort utilization via nvidia-smi
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=1.0,
        )
        first_line = output.strip().splitlines()[0]
        stats["gpu/utilization_pct"] = float(first_line)
    except Exception:
        pass

    return stats
