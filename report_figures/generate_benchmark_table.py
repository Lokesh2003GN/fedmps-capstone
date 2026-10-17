#!/usr/bin/env python3
"""Benchmark comparison table (results/benchmark_comparison_table.md).

Reads results/benchmark_measured.json and
results/experiments_vgg_big_threeway.json and tabulates exactly what they
contain. A value that was not measured is written as "not measured", and an
operating point the sample size cannot resolve as "not resolvable".

Usage:
    python report_figures/compute_benchmark_data.py    # measure first
    python report_figures/generate_benchmark_table.py  # then tabulate
"""
import sys as _sys

for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # project root
RESULTS = ROOT / "results"

# Rows of Table 1, in order: key in benchmark_measured.json -> label.
SERIES = {
    "clean":               "Clean probes (upper bound)",
    "degraded":            "CCTV degraded (frozen backbone)",
    "degraded_calibrated": "+ scoring layer",
    "federated_adapter":   "+ federated adapter",
}


def log(msg: str) -> None:
    print(f"[table] {msg}", flush=True)


def load():
    p = RESULTS / "benchmark_measured.json"
    if not p.exists():
        raise SystemExit(
            f"{p.name} not found. Run:  python report_figures/compute_benchmark_data.py")
    return json.loads(p.read_text())


def fmt(v, nd: int = 4) -> str:
    """Render a measured value, or an explicit dash when it is not one."""
    if v is None:
        return "not measured"
    if isinstance(v, float) and math.isnan(v):
        return "not resolvable"
    return f"{v:.{nd}f}"


def table(data) -> Path:
    arms, proto = data["arms"], data["protocol"]
    rows = []
    for key, label in SERIES.items():
        if key not in arms:
            continue
        a = arms[key]
        cmc, ver, tpir = a.get("cmc", {}), a["verification"], a["tpir"]
        rows.append((label,
                     fmt(ver.get("tar@far=0.01")), fmt(ver.get("tar@far=0.001")),
                     fmt(ver.get("tar@far=0.0001")),
                     fmt(tpir.get("tpir@fpir=0.01")),
                     fmt(tpir.get("tpir@fpir=0.001")),
                     fmt(cmc.get("rank1")), fmt(cmc.get("rank5")),
                     fmt(cmc.get("rank10"))))

    head = ("| Arm | TAR@FAR=1e-2 | TAR@FAR=1e-3 | TAR@FAR=1e-4 | "
            "TPIR@FPIR=1e-2 | TPIR@FPIR=1e-3 | Rank-1 | Rank-5 | Rank-10 |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    body = "".join("| " + " | ".join(r) + " |\n" for r in rows)

    three = ""
    p = RESULTS / "experiments_vgg_big_threeway.json"
    if p.exists():
        tw = json.loads(p.read_text())["three_way"]

        def cmc_of(arm):
            return tw[arm]["identification"]["cmc"]

        # local_only is per node; report the mean of the five nodes' CMCs.
        nodes = [n["identification"]["cmc"]
                 for n in tw["local_only"]["per_node"].values()]
        local = {k: float(np.mean([c[k] for c in nodes])) for k in nodes[0]}
        arms3 = [("Frozen backbone, no adapter", cmc_of("baseline_no_adapter")),
                 ("Local only (mean of 5 nodes)", local),
                 ("Federated (FedMPS)", cmc_of("federated")),
                 ("Centralised (upper bound)", cmc_of("centralised"))]
        three = (
            "\n### Table 2: federated learning against its bounds\n\n"
            "Closed-set identification on the same held-out identities, from "
            "the three-way run (`results/experiments_vgg_big_threeway.json`).\n\n"
            "| Arm | Rank-1 | Rank-5 | Rank-10 |\n|---|---|---|---|\n"
            + "".join(f"| {name} | {fmt(c.get('rank1'))} | {fmt(c.get('rank5'))} "
                      f"| {fmt(c.get('rank10'))} |\n" for name, c in arms3)
        )

    md = f"""# FedMPS capstone benchmark results

**Every number below is measured.** Cells reading "not resolvable" are
operating points the sample size cannot support, and the code returns NaN for
them rather than a number; cells reading "not measured" are quantities that
run did not compute. Neither is filled in by estimation.

Protocol: **{proto['dataset']}**, {proto['n_identities']:,} identities,
{proto['n_images']:,} images, ArcFace-aligned
({proto['alignment']['aligned']:,}/{proto['alignment']['aligned'] + proto['alignment']['missed']:,}
faces aligned). Gallery {proto['n_gallery']:,} identities;
{proto['n_mate_probes']:,} mate probes; {proto['n_nonmate_probes']:,} non-mate
probes; {proto['n_verification_pairs']:,} verification pairs.
Operating point: {proto['n_enrol']} enrolment photo(s) x
{proto['n_probe_frames']} camera frame(s). Backbone `{proto['backbone']}`, frozen.

Resolution limits: the finest measurable FPIR is
{proto['finest_fpir']:.2e} ({proto['n_nonmate_probes']:,} non-mates) and the
finest FAR is {proto['finest_far']:.2e}
({proto['n_verification_pairs']:,} pairs). TAR@FAR=1e-4 is therefore below
what this verification set can resolve, which is why it reports as
not resolvable rather than as a value.

---

### Table 1: measured across the surveillance pipeline

{head}{body}
{three}
---

### Reproducing

```bash
python report_figures/compute_benchmark_data.py    # measures, writes results/benchmark_measured.json
python report_figures/generate_benchmark_table.py  # tabulates only what the result files contain
```
"""
    out = RESULTS / "benchmark_comparison_table.md"
    out.write_text(md, encoding="utf-8")
    log(f"wrote {out.name}")
    return out


def main() -> int:
    data = load()
    log(f"arms measured: {', '.join(data['arms'])}")
    table(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
