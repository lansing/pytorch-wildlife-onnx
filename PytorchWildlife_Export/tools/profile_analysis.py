#!/usr/bin/env python3
"""
ORT profiling JSON post-processor for quantization analysis.

Usage:
    python profile_analysis.py profile.json [profile2.json ...] [options]

Options:
    --warmup-runs N     Number of warmup runs recorded before measurement (default: 0)
    --total-runs N      Total runs in the file (warmup + measurement).
                        Required when --warmup-runs > 0.
    --top N             Show top-N slow nodes (default: 20)
    --min-pct FLOAT     Hide op-type rollup rows below this %% of total time (default: 0.1)
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dtype_of_shape(shape_list: list) -> set:
    """
    Extract dtype keys from an ORT input_type_shape / output_type_shape list.
    Each element is a dict like {"float": [1, 16, 320, 320]} or {"int8": [...]}.
    """
    dtypes = set()
    for item in (shape_list or []):
        if isinstance(item, dict):
            dtypes.update(item.keys())
    return dtypes


def _is_int8_kernel(event: dict) -> bool:
    """True if any input OR output tensor is int8 — indicates an INT8 fused kernel."""
    args = event.get("args", {})
    dtypes = (
        _dtype_of_shape(args.get("input_type_shape", []))
        | _dtype_of_shape(args.get("output_type_shape", []))
    )
    return "int8" in dtypes


def _is_qdq_overhead(op_name: str) -> bool:
    return op_name in ("QuantizeLinear", "DequantizeLinear")


def _is_memcpy(op_name: str) -> bool:
    return "memcpy" in op_name.lower()


def _is_trt_kernel(op_name: str) -> bool:
    """TRT EP compiles the whole (sub)graph into a single fused engine call."""
    return op_name.startswith("TRTKernel")


def _quant_class(event: dict, op_name: str) -> str:
    if _is_memcpy(op_name):
        return "MemcpyHost↔Device"
    if _is_qdq_overhead(op_name):
        return "Q/DQ overhead"
    if _is_trt_kernel(op_name):
        return "TRT-fused subgraph"
    if _is_int8_kernel(event):
        return "INT8 kernel"
    return "Float kernel"


def _strip_suffix(name: str) -> str:
    for suffix in ("_kernel_time", "_fence_before", "_fence_after", "_memcpy"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _short_name(full_name: str, width: int = 55) -> str:
    """
    Shorten a YOLO node path for display while preserving enough context to be
    unambiguous. Keeps the last 4 slash-separated segments; if still longer than
    `width`, truncates from the left with an ellipsis.
    """
    parts = full_name.split("/")
    short = "/".join(parts[-4:]) if len(parts) > 4 else full_name
    if len(short) > width:
        short = "…" + short[-(width - 1):]
    return short


# ---------------------------------------------------------------------------
# Loading + warmup filtering
# ---------------------------------------------------------------------------

def load_events(path: str, warmup_runs: int = 0, total_runs: int = 1) -> list:
    """
    Load node kernel events from an ORT profiling JSON file.

    When warmup_runs > 0, the first (warmup_runs / total_runs) fraction of
    events (by timestamp order) is discarded so warmup overhead is excluded
    from analysis. This works because ORT emits a fixed number of events per
    inference run for a given graph.
    """
    with open(path) as f:
        data = json.load(f)

    events = [
        ev for ev in data
        if ev.get("cat") == "Node" and ev["name"].endswith("_kernel_time")
    ]

    if not events:
        print(f"  WARNING: no kernel_time node events found in {path}", file=sys.stderr)
        return []

    events.sort(key=lambda e: e["ts"])

    if warmup_runs > 0 and total_runs > warmup_runs:
        events_per_run = len(events) // total_runs
        skip = warmup_runs * events_per_run
        print(
            f"  Skipping {skip} warmup events "
            f"({warmup_runs} runs × {events_per_run} events/run), "
            f"{len(events) - skip} measurement events remain"
        )
        events = events[skip:]

    return events


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(events: list) -> dict:
    by_op: dict = defaultdict(lambda: {"count": 0, "total_us": 0})
    by_provider: dict = defaultdict(lambda: {"count": 0, "total_us": 0})
    by_quant: dict = defaultdict(lambda: {"count": 0, "total_us": 0})
    # Separate CPU-provider ops (they trigger MemcpyFromHost and add synchronisation cost)
    cpu_ops: list = []
    # Per-node accumulation: key = (full_stripped_name, op_name, provider, quant_class)
    node_acc: dict = defaultdict(lambda: {"total_us": 0, "count": 0})

    for ev in events:
        args = ev.get("args", {})
        dur = ev.get("dur", 0)
        op_name = args.get("op_name", ev["name"])
        provider = args.get("provider", "unknown")
        qclass = _quant_class(ev, op_name)
        node_key = (_strip_suffix(ev["name"]), op_name, provider, qclass)

        by_op[op_name]["count"] += 1
        by_op[op_name]["total_us"] += dur

        by_provider[provider]["count"] += 1
        by_provider[provider]["total_us"] += dur

        by_quant[qclass]["count"] += 1
        by_quant[qclass]["total_us"] += dur

        node_acc[node_key]["total_us"] += dur
        node_acc[node_key]["count"] += 1

        if provider == "CPUExecutionProvider":
            cpu_ops.append((ev["name"], op_name, dur))

    total_us = sum(ev.get("dur", 0) for ev in events)

    slow_nodes = sorted(
        [
            (name, op, prov, qcls, acc["total_us"], acc["count"])
            for (name, op, prov, qcls), acc in node_acc.items()
        ],
        key=lambda r: r[4],
        reverse=True,
    )

    return {
        "by_op": dict(by_op),
        "by_provider": dict(by_provider),
        "by_quant": dict(by_quant),
        "slow_nodes": slow_nodes,
        "cpu_ops": cpu_ops,
        "total_us": total_us,
        "total_events": len(events),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _pct(part, total):
    return 100.0 * part / total if total else 0.0


def _ms(us: float) -> str:
    return f"{us / 1000:.3f}"


PROV_SHORT = {
    "CUDAExecutionProvider": "CUDA",
    "CPUExecutionProvider": "CPU ",
    "TensorrtExecutionProvider": "TRT ",
}

W = 90  # report width


def print_report(path: str, events: list, result: dict, top_n: int, min_pct: float):
    total_us = result["total_us"]
    total_ms = total_us / 1000.0
    by_quant = result["by_quant"]
    n_int8 = by_quant.get("INT8 kernel", {}).get("count", 0)
    n_trt = by_quant.get("TRT-fused subgraph", {}).get("count", 0)
    n_qdq = by_quant.get("Q/DQ overhead", {}).get("count", 0)
    n_memcpy = by_quant.get("MemcpyHost↔Device", {}).get("count", 0)

    print()
    print("=" * W)
    print(f"  ORT Profile Analysis: {Path(path).name}")
    print(f"  Events analysed : {len(events):,}   |   Total kernel time: {total_ms:.2f} ms")
    print(f"  TRT subgraphs: {n_trt}   |   INT8 kernels: {n_int8}   |   "
          f"Q/DQ nodes: {n_qdq}   |   MemcpyFromHost: {n_memcpy}")

    if n_trt > 0:
        pct_trt = _pct(by_quant["TRT-fused subgraph"]["total_us"], total_us)
        print(f"  ✓  {pct_trt:.1f}% of kernel time runs inside TRT-compiled engine(s).")
        print(f"     Internal layer precision is not visible in ORT profiling.")
        print(f"     Re-run with trt_profile_verbosity=verbose for per-layer TRT timing.")
    elif n_qdq > 0 and n_int8 == 0:
        print()
        print("  ⚠  QDQ NODES PRESENT BUT NO INT8 KERNELS FUSED.")
        print("     ORT CUDA EP is not fusing QDQ→Conv into INT8 kernels.")
        print("     All compute is running float32 with pure QDQ overhead.")
        print("     Consider: TensorRT EP, or ORT INT8 kernel fusion prerequisites.")
    elif n_int8 > 0:
        pct_int8 = _pct(by_quant["INT8 kernel"]["total_us"], total_us)
        print(f"  ✓  {pct_int8:.1f}% of kernel time runs as INT8.")

    print("=" * W)

    sep = "─" * W

    # ── By Op Type ──────────────────────────────────────────────────────────
    print(f"\n── By Op Type (>= {min_pct}% of total) " + "─" * (W - 34 - len(f"{min_pct}")))
    print(f"  {'Op':<32} {'Count':>7}  {'Total (ms)':>12}  {'Avg (μs)':>10}  {'%':>6}")
    print(f"  {sep}")
    rows = sorted(result["by_op"].items(), key=lambda x: x[1]["total_us"], reverse=True)
    hidden = 0
    for op, s in rows:
        pct = _pct(s["total_us"], total_us)
        if pct < min_pct:
            hidden += 1
            continue
        avg_us = s["total_us"] / s["count"] if s["count"] else 0
        print(
            f"  {op:<32} {s['count']:>7}  {_ms(s['total_us']):>12}  {avg_us:>10.1f}  {pct:>5.1f}%"
        )
    if hidden:
        print(f"  ({hidden} op types below {min_pct}% threshold omitted)")

    # ── By Provider ─────────────────────────────────────────────────────────
    print(f"\n── By Provider " + "─" * (W - 15))
    print(f"  {'Provider':<40} {'Count':>7}  {'Total (ms)':>12}  {'%':>6}")
    print(f"  {sep}")
    rows = sorted(result["by_provider"].items(), key=lambda x: x[1]["total_us"], reverse=True)
    for prov, s in rows:
        pct = _pct(s["total_us"], total_us)
        print(f"  {prov:<40} {s['count']:>7}  {_ms(s['total_us']):>12}  {pct:>5.1f}%")

    # ── By Quantization Class ────────────────────────────────────────────────
    print(f"\n── By Quantization Mode " + "─" * (W - 24))
    print(f"  {'Mode':<24} {'Count':>7}  {'Total (ms)':>12}  {'%':>6}")
    print(f"  {sep}")
    order = ["TRT-fused subgraph", "INT8 kernel", "Float kernel", "Q/DQ overhead", "MemcpyHost↔Device"]
    for cls in order + [k for k in by_quant if k not in order]:
        if cls not in by_quant:
            continue
        s = by_quant[cls]
        pct = _pct(s["total_us"], total_us)
        print(f"  {cls:<24} {s['count']:>7}  {_ms(s['total_us']):>12}  {pct:>5.1f}%")

    # ── CPU-executed nodes (source of MemcpyFromHost overhead) ───────────────
    cpu_ops = result["cpu_ops"]
    if cpu_ops:
        print(f"\n── CPU-Executed Nodes ({len(cpu_ops)} — each triggers host→device sync) " + "─" * 10)
        # Group by op_name
        cpu_by_op: dict = defaultdict(lambda: {"count": 0, "total_us": 0})
        for _, op_name, dur in cpu_ops:
            cpu_by_op[op_name]["count"] += 1
            cpu_by_op[op_name]["total_us"] += dur
        print(f"  {'Op':<32} {'Count':>7}  {'Total (ms)':>12}  {'Avg (μs)':>10}")
        print(f"  {sep}")
        for op, s in sorted(cpu_by_op.items(), key=lambda x: x[1]["total_us"], reverse=True):
            avg_us = s["total_us"] / s["count"] if s["count"] else 0
            print(f"  {op:<32} {s['count']:>7}  {_ms(s['total_us']):>12}  {avg_us:>10.1f}")

    # ── Slow Nodes ───────────────────────────────────────────────────────────
    NW = 56  # node name column width
    print(f"\n── Top {top_n} Slow Nodes " + "─" * (W - 18 - len(str(top_n))))
    print(
        f"  {'Node':<{NW}} {'Op':<22} {'P':>4}  {'Mode':<5}  "
        f"{'Total (ms)':>10}  {'Avg (μs)':>9}  {'%':>6}"
    )
    print(f"  {sep}")

    for name, op, prov, qcls, tot_us, count in result["slow_nodes"][:top_n]:
        pct = _pct(tot_us, total_us)
        avg_us = tot_us / count if count else 0
        short_prov = PROV_SHORT.get(prov, prov[:4])
        if qcls == "TRT-fused subgraph":
            mode_flag = "TRT"
        elif qcls == "INT8 kernel":
            mode_flag = "INT8"
        elif _is_qdq_overhead(op):
            mode_flag = "QDQ"
        elif _is_memcpy(op):
            mode_flag = "COPY"
        else:
            mode_flag = "fp32"
        display_name = _short_name(name, NW)
        print(
            f"  {display_name:<{NW}} {op:<22} {short_prov}  {mode_flag:<5}  "
            f"{_ms(tot_us):>10}  {avg_us:>9.1f}  {pct:>5.1f}%"
        )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Post-process ORT profiling JSON for quantization analysis."
    )
    parser.add_argument("profiles", nargs="+", help="ORT profiling JSON file(s)")
    parser.add_argument(
        "--warmup-runs", type=int, default=0,
        help="Number of warmup runs to exclude from analysis (default: 0)"
    )
    parser.add_argument(
        "--total-runs", type=int, default=None,
        help="Total runs recorded in the file (warmup + measurement). "
             "Required when --warmup-runs > 0."
    )
    parser.add_argument(
        "--top", type=int, default=20,
        help="Number of slow nodes to show (default: 20)"
    )
    parser.add_argument(
        "--min-pct", type=float, default=0.1,
        help="Hide op-type rows below this %% of total time (default: 0.1)"
    )
    args = parser.parse_args()

    if args.warmup_runs > 0 and args.total_runs is None:
        parser.error("--total-runs is required when --warmup-runs > 0")

    total_runs = args.total_runs or 1

    for path in args.profiles:
        print(f"\nLoading: {path}")
        events = load_events(path, warmup_runs=args.warmup_runs, total_runs=total_runs)
        if not events:
            continue
        result = analyse(events)
        print_report(path, events, result, top_n=args.top, min_pct=args.min_pct)


if __name__ == "__main__":
    main()
