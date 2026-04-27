# WSInsight regression suite

Per-slide tests that compare the current WSInsight build against reference
outputs from a known-good revision. They are **opt-in** — the suite skips
silently when no slides are configured, so adding it to CI is free.

## Three tiers

| Tier | What it checks | Speed | Marker |
|---|---|---|---|
| 1. metadata | MPP, AppMag, AppMag-fallback firing for slides that need it | seconds | `regression` |
| 2. patches | `coords` HDF5 produced by `wsinsight patch` matches a golden file | minutes | `regression`, `slow` |
| 3. inference | per-slide model output CSV matches a golden file (numeric tol) | minutes-hours | `regression`, `slow` |

All three tiers iterate over the same case list in
[tests/regression/cases.toml](cases.toml).

## Configure your slides

Edit `cases.toml` (or point `WSINSIGHT_REGRESSION_CASES` at your own copy):

```toml
[[case]]
slide_id = "TCGA-D7-6818"
path     = "/workspace/tcga-stad/dataset/.../TCGA-D7-6818-...svs"
expected_appmag = 40.0
expected_mpp    = 0.252

[[case]]
slide_id = "TCGA-CG-5727"
path     = "/workspace/tcga-stad/dataset/.../TCGA-CG-5727-...svs"
expected_appmag = 40.0
expected_mpp    = 0.25
expects_appmag_fallback = true   # this slide must trigger the fallback warning
```

`slide_id` is used both as the test id and as the folder name under
`fixtures/<slide_id>/`. Cases whose `path` does not exist are skipped.

## Generate baselines (run on the "good" build)

```bash
git checkout <known-good-tag>
python scripts/regenerate_regression_baselines.py \
    --zoo-model /app/zoo/huangch/CellViT-SAM-H-x40/main \
    --workdir /tmp/wsinsight-baseline
git checkout -    # back to your branch
git add tests/regression/fixtures
git commit -m "regression: refresh baselines from <tag>"
```

This populates each case's `fixtures/<slide_id>/` with `patches.h5` and
`inference.csv`.

## Run the suite (on the build under test)

```bash
# Tier 1 only — fast, no model needed:
pytest tests/regression -m regression

# Tiers 2 + 3 — re-run patch + infer in-process, compare to goldens:
pytest tests/regression -m regression --run-slow \
    --zoo-model /app/zoo/huangch/CellViT-SAM-H-x40/main

# Compare an existing run dir without re-running the pipeline:
pytest tests/regression -m regression \
    --patch-output-dir /workspace/tcga-stad/results/run-2026-04-26 \
    --infer-output-dir /workspace/tcga-stad/results/run-2026-04-26
```

## Tolerances

- MPP: per-case `mpp_atol` (default 0.01).
- AppMag: ±0.5.
- Patch coordinates: exact set equality (count + (x, y) pairs), order-insensitive.
- Inference numeric columns: `rtol=1e-3, atol=1e-4`.
- Inference non-numeric columns: exact match.

Adjust thresholds inside the test files if you intentionally change scoring
internals.
