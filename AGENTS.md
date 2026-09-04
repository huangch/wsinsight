# WSInsight — Agent Guide

Whole-slide-image (WSI) inference pipeline: `wsinsight` CLI (click) → cell detections + model outputs (H5/CSV/GeoJSON). Python >=3.8.

## Environment (read first)

- **Never use system `python`** — it is Python 2.7. Always use a conda env.
- Conda base: `/opt/anaconda3` (base env only by default; no `wsinsight` env on this host yet).
- Create env: `bash ./conda-setup.sh wsinsight --mcp` (the `-m`/`--mcp` flag installs fastmcp; it is **not** installed by default to avoid jaraco version-scanning issues). Add `-d`/`--dev` to also install pytest/pytest-cov/ruff/pre_commit for running the test suite; add `-r`/`--reset` to nuke and recreate the env. Run `./conda-setup.sh --help` for the full CLI.
- Or run everything in Docker: `wsinsight:latest` (see below) — this is the primary runtime.
- Known install workarounds live in `conda-setup.sh` (histomicstk `--no-deps`, explicit `large-image`, pyvips SSL fallback, `PIP_CACHE_DIR=/tmp` for NAS inode quotas). Don't "fix" them.

## Docker

- `./wsinsight-docker-run.sh [DATA_DIR] [COMMAND...]` — wrapper for `docker run` (options: `--gpu <id>`, `--tmpdir <dir>`). HF model cache persists in named volume `wsinsight-hf-cache`.
- The image **starts as root by design**: `docker-entrypoint.sh` remaps the baked-in `user` (uid 1000) to the owner of the mounted `/workspace` (or `$HOST_UID`/`$HOST_GID`), then drops privileges via `setpriv`.
- **Gotcha:** if the mounted dir is root-owned and `HOST_UID`/`HOST_GID` are unset, the session runs as root. Fix: `export HOST_UID=1000 HOST_GID=1000` or mount a dir you own.
- Build: `./docker-build-push.sh`; entrypoint logic: `docker-entrypoint.sh`.

## MCP server (`wsinsight-mcp`)

- Entry point `wsinsight.mcp.__main__:main`; extra `mcp = ["fastmcp>=2.0"]`. stdio by default (works with VS Code Copilot); `--http HOST:PORT` (default 8765), `--experimental`, `--max-concurrent`.
- Tools are auto-registered from the **bundled** `wsinsight/cli/cli_schema.json` (14 commands, `schema_version: 1`). The live generator is `wsinsight schema` — the bundle is NOT regenerated at runtime. If you change CLI params, regenerate the bundle and keep `tests/test_mcp_pkg/test_mcp.py` parity passing (it is command-level only, not param-freshness).
- By default only STABLE commands are exposed: `run, patch, infer, ncomp, export, reg`. `--experimental` adds: `hplot, hplot-finalize, ecomp, tcomp, niche, niche-profile, agg, import`.
- Long-running commands (`run, patch, infer, ncomp, hplot, ecomp, tcomp, niche, agg`) return a `job_id`; clients poll `job_status`/`job_logs`/`cancel_job`.
- The server reports `wsinsight.__version__` in its MCP `serverInfo`; without an explicit `version=` FastMCP would report its own.
- Adapter (`wsinsight/mcp/adapters.py`) translates snake_case args → kebab-case `--flags` only. **No positional args are supported** — keep all CLI params flag-based.

## Commands (14)

`run` (main pipeline), `patch`, `export`, `infer`, `ncomp`, `reg`, `hplot`, `hplot-finalize`, `ecomp`, `tcomp`, `niche`, `niche-profile`, `agg`, `import`. CLI source: `wsinsight/cli/cli.py`. All require `--results-dir`.

- `--wsi-dir` belongs only to the stages that read slide pixels: `run`, `patch`, `infer`, `reg`, `import`. The analysis stages (`ncomp`, `ecomp`, `tcomp`, `agg`, `hplot`, `niche`, `export`, `hplot-finalize`, `niche-profile`) **reject** it and derive the slide list from `patches/` under `--results-dir`.
- `--spacing-um-px` on `run`/`patch` **overrides** the slide metadata (`0` = use metadata, error if the slide has none). `patch` records the spacing in `patches/<slide>.h5`; downstream stages read it back via `insight_helpers.build_slide_mpp_lookup`, so one flag governs the pipeline. `reg`'s same-named option is a fallback instead — don't unify them without checking `tests/test_spacing_contract.py`.

## Tests & CI

- Run: `python -m pytest tests/` (CI: `--verbose`, Python 3.8–3.12 matrix in `.github/workflows/ci.yml`; Docker job runs `tests/test_all.py`).
- Lint: ruff (`.github/workflows/ruff.yml`, config in `pyproject.toml`); mypy config also in `pyproject.toml`.

## Sibling repos (same ecosystem)

- `hplot` — H-Plot stats/plotting core (border-layer inference: cluster-mass permutation, GAM); hplot CLI commands in wsinsight call into it.
- `sptxinsight` — spatial-transcriptomics sibling (AnnData in, micron coords); vendored H-Plot engine; own MCP on port 8766.
- `clawsight` / `clawpyter` — client-side agent plugins driving these MCP servers / Jupyter.
