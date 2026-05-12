import os

from better_ai.monitoring.observability import (
    ObservabilityAdapter,
    ObservabilityConfig,
    NoOpObservabilityBackend,
    collect_gpu_stats,
)


def test_noop_backend_when_disabled():
    adapter = ObservabilityAdapter(ObservabilityConfig(backend="none"))
    assert isinstance(adapter.backend, NoOpObservabilityBackend)
    adapter.start_run(config={"a": 1})
    adapter.log_metrics({"train/loss": 1.0}, step=1)
    adapter.log_artifact("/tmp/no-file")
    adapter.finish_run(status="completed")


def test_metrics_payload_shape_contains_expected_keys():
    stats = collect_gpu_stats()
    assert "gpu/memory_allocated_gb" in stats
    assert "gpu/memory_reserved_gb" in stats
    assert "gpu/utilization_pct" in stats or os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def test_from_config_env_disabled_defaults_to_noop(monkeypatch):
    monkeypatch.setenv("BETTER_AI_OBSERVABILITY_BACKEND", "none")

    class Cfg:
        pass

    adapter = ObservabilityAdapter.from_config(Cfg())
    assert isinstance(adapter.backend, NoOpObservabilityBackend)
