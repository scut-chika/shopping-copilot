"""Measure the index's real memory footprint and build time.

The rules reserve the right to score under memory and timeout restrictions, so
this is a resource-envelope check rather than a micro-benchmark.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rss_mb() -> tuple[float, float]:
    """Current and peak working set, in MB. Returns (0, 0) off Windows."""
    if not sys.platform.startswith("win"):
        try:
            import resource
            return 0.0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return 0.0, 0.0

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)] + [
            (n, ctypes.c_size_t) for n in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")
        ]

    counters = PMC()
    counters.cb = ctypes.sizeof(PMC)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess())
    for lib, name in ((ctypes.windll.kernel32, "K32GetProcessMemoryInfo"),
                      (ctypes.windll.psapi, "GetProcessMemoryInfo")):
        fn = getattr(lib, name, None)
        if fn is None:
            continue
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_uint32]
        fn.restype = ctypes.c_int
        if fn(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / 1e6, counters.PeakWorkingSetSize / 1e6
    return 0.0, 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Index memory and build-time check")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output", default="memory_profile.json")
    args = parser.parse_args()

    before, _ = rss_mb()
    from copilot.catalog import CatalogIndex
    from starter.agent import _config_from_env

    config = _config_from_env()
    started = time.perf_counter()
    index = CatalogIndex(args.catalog, config)
    build = time.perf_counter() - started
    current, peak = rss_mb()

    # tracemalloc is measured in a second pass: it multiplies allocation cost,
    # so timing it would report a build time nobody will ever see.
    tracemalloc.start()
    CatalogIndex(args.catalog, config)
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    report = {
        "config": {
            "retain_products": config.retain_products,
            "use_loose_index": config.use_loose_index,
            "use_profile": config.use_profile,
            "use_constraint_mining": config.use_constraint_mining,
        },
        "build_seconds": round(build, 2),
        "rss_before_mb": round(before, 1),
        "rss_after_mb": round(current, 1),
        "rss_peak_mb": round(peak, 1),
        "index_growth_mb": round(current - before, 1),
        "traced_peak_mb": round(traced_peak / 1e6, 1),
        "products_indexed": len(index),
        "distinct_constraints": len(index.card_index),
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
