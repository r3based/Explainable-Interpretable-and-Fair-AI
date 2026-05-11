"""Runtime benchmarking helpers for explanation methods."""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable

import torch


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def measure_runtime(
    fn: Callable[..., Any],
    *args,
    repeats: int = 5,
    warmup: int = 1,
    device: str | torch.device | None = None,
    measure_peak_memory: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Measure runtime for a callable while excluding process startup costs."""
    runtime_device = _resolve_device(device)

    for _ in range(int(warmup)):
        _maybe_sync(runtime_device)
        fn(*args, **kwargs)
        _maybe_sync(runtime_device)

    timings: list[float] = []
    peak_memory_mb = 0.0
    for _ in range(int(repeats)):
        if runtime_device.type == "cuda" and torch.cuda.is_available() and measure_peak_memory:
            torch.cuda.reset_peak_memory_stats(runtime_device)

        _maybe_sync(runtime_device)
        started = time.perf_counter()
        fn(*args, **kwargs)
        _maybe_sync(runtime_device)
        elapsed = time.perf_counter() - started
        timings.append(float(elapsed))

        if runtime_device.type == "cuda" and torch.cuda.is_available() and measure_peak_memory:
            peak_memory_mb = max(
                peak_memory_mb,
                float(torch.cuda.max_memory_allocated(runtime_device) / (1024 ** 2)),
            )

    return {
        "per_run_seconds": timings,
        "runtime_mean_sec": float(statistics.mean(timings)),
        "runtime_std_sec": float(statistics.pstdev(timings) if len(timings) > 1 else 0.0),
        "runtime_median_sec": float(statistics.median(timings)),
        "repeats": int(repeats),
        "warmup": int(warmup),
        "device": str(runtime_device),
        "peak_memory_mb": peak_memory_mb,
    }


def benchmark_explainer(
    method_name: str,
    callables: list[Callable[[], Any]],
    repeats: int = 1,
    warmup: int = 1,
    device: str | torch.device | None = None,
    measure_peak_memory: bool = True,
    log_progress: bool = False,
) -> dict[str, Any]:
    """Benchmark per-image explanation callables for one method."""
    if not callables:
        raise ValueError("callables must contain at least one explanation function.")

    per_image_runtime_sec: list[float] = []
    peak_memory_mb = 0.0
    resolved_device = _resolve_device(device)

    for idx, callable_fn in enumerate(callables, start=1):
        if log_progress:
            print(f"runtime {method_name}: image {idx}/{len(callables)}", flush=True)
        result = measure_runtime(
            callable_fn,
            repeats=repeats,
            warmup=warmup,
            device=resolved_device,
            measure_peak_memory=measure_peak_memory,
        )
        per_image_runtime_sec.append(float(result["runtime_mean_sec"]))
        peak_memory_mb = max(peak_memory_mb, float(result["peak_memory_mb"]))
        if log_progress:
            print(
                f"runtime {method_name}: image {idx}/{len(callables)} done "
                f"mean={result['runtime_mean_sec']:.2f}s",
                flush=True,
            )

    return {
        "method": method_name,
        "device": str(resolved_device),
        "num_images": len(callables),
        "per_image_runtime_sec": per_image_runtime_sec,
        "runtime_mean_sec": float(statistics.mean(per_image_runtime_sec)),
        "runtime_std_sec": float(statistics.pstdev(per_image_runtime_sec) if len(per_image_runtime_sec) > 1 else 0.0),
        "runtime_median_sec": float(statistics.median(per_image_runtime_sec)),
        "peak_memory_mb": peak_memory_mb,
        "benchmark_repeats": int(repeats),
        "warmup": int(warmup),
    }


def benchmark_all_methods(
    method_callables: dict[str, list[Callable[[], Any]]],
    repeats: int = 1,
    warmup: int = 1,
    device: str | torch.device | None = None,
    measure_peak_memory: bool = True,
    log_progress: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for method_name, callables in method_callables.items():
        if log_progress:
            print(f"Starting runtime benchmark for {method_name} ({len(callables)} images)", flush=True)
        results.append(
            benchmark_explainer(
                method_name=method_name,
                callables=callables,
                repeats=repeats,
                warmup=warmup,
                device=device,
                measure_peak_memory=measure_peak_memory,
                log_progress=log_progress,
            )
        )
    return results
