# WSInsight — Agent Guide

Whole-slide-image (WSI) inference pipeline: `wsinsight` CLI (click) → cell detections + model outputs (H5/CSV/GeoJSON). Python >=3.8.

## Environment (read first)

- Docker (`wsinsight:latest`, see below) is the **primary runtime** — prefer it unless the task requires a native env (e.g. editable installs, debugging, running the test matrix).
- When running natively instead of in Docker: **never use system `python`** — it is Python 2.7. Always use a conda env.
- Conda base: `/opt/anaconda3` (base env only by default; no `wsinsight` env on this host yet).
- Before running any `wsinsight` command via conda, verify the env exists with `conda env list`; if absent, create it first with `bash ./conda-setup.sh wsinsight` before proceeding.
- Create env: `bash ./conda-setup.sh wsinsight --mcp`. Use `--mcp` (short form `-m`) to install fastmcp; it is **not** installed by default to avoid jaraco version-scanning issues. Add `--dev` (short form `-d`) to also install pytest/pytest-cov/ruff/pre_commit for running the test suite; add `--reset` (short form `-r`) to nuke and recreate the env. Run `./conda-setup.sh --help` for the full CLI.
- Known install workarounds live in `conda-setup.sh` (histomicstk `--no-deps`, explicit `large-image`, pyvips SSL fallback, `PIP_CACHE_DIR=/tmp` for NAS inode quotas). Don't "fix" them.

## Running WSInsight (the unified wrapper)

- `./wsinsight.sh` is the **single entry point** for running `wsinsight`. It manages BOTH backends — `native` (the wsinsight CLI on the host inside the activated conda env) and `docker` (the `huangchtw/wsinsight:latest` container). The legacy `wsinsight.sh` env-trampoline and the legacy `wsinsight-docker-run.sh` wrapper have both moved to `bak_old_scripts/`.
- Subcommands: `./wsinsight.sh run [-b native|docker] [--gpu ID|all] [--tmpdir DIR] [--no-pull] [--dry-run] [WSINSIGHT_ARGS ...]`, `./wsinsight.sh status`, `./wsinsight.sh doctor [-b ...]`, `./wsinsight.sh where`. Run `./wsinsight.sh --help` for the full surface.
- **Param-parsing rule**: everything before the first wsinsight subcommand name (`run`, `patch`, `infer`, ...) is consumed by the wrapper (env control: `-b`, `--gpu`, `--tmpdir`, `--no-pull`, `--dry-run`). From (and including) the first wsinsight subcommand name onward, every token is passed through verbatim. Use `--` to force passthrough explicitly.
- **Default backend**: `native`. Override with `-b docker`, or set `WSINSIGHT_BACKEND=docker` in the environment.
- **Discovery of wsinsight subcommands** (for param parsing): cached at `$HOME/.cache/wsinsight/commands.txt` (TTL `WSINSIGHT_COMMANDS_TTL_SECONDS`, default 86400) via `wsinsight schema --commands-only`. Falls back to a static builtin list if wsinsight isn't on PATH.
- **Single-shard workflows use `./wsinsight.sh run`**. **Multi-shard parallel workflows keep using `./tmux-multi-gpu.sh`** (it intentionally does its own per-shard invocation; do not unify).

## Docker

- The image `huangchtw/wsinsight:latest` **starts as root by design**: `docker-entrypoint.sh` remaps the baked-in `user` (uid 1000) to `$HOST_UID`/`$HOST_GID` if set, otherwise to the owner of the mounted `/workspace`, otherwise to `1000:1000`; then drops privileges via `setpriv`.
- **Gotcha:** if the mounted dir is root-owned and `HOST_UID`/`HOST_GID` are unset, the docker session runs as root. Fix: `export HOST_UID=1000 HOST_GID=1000` or mount a dir you own.
- You should rarely need to invoke `docker run` directly — use `./wsinsight.sh -b docker run ...` instead. The wrapper honors `--gpu <id>|all`, `--tmpdir <dir>`, `--no-pull`, and forwards `HOST_UID`/`HOST_GID` to the container when set.
- Build: `./docker-build-push.sh`; entrypoint logic: `docker-entrypoint.sh`.

## MCP server (`wsinsight-mcp`)

- Entry point `wsinsight.mcp.__main__:main`; extra `mcp = ["fastmcp>=4.0,<5"]`. stdio by default (works with VS Code Copilot); `--http HOST:PORT` (default 8765), `--experimental`, `--max-concurrent`.
- Tools are auto-registered from the **bundled** `wsinsight/cli/cli_schema.json` (14 commands, `schema_version: 1`).
- The *command surface* is NOT regenerated at runtime; `wsinsight/mcp/schema.py::load_schema` only overlays the live model zoo on top (the `models` list and `model_name` choices) so `WSINSIGHT_ZOO_REGISTRY_PATH` is honoured.
- After changing CLI params, regenerate the bundle:
  1. Run `wsinsight schema --output wsinsight/cli/cli_schema.json` with no local zoo registry resolvable (unset `WSINSIGHT_ZOO_REGISTRY_PATH`), otherwise machine-specific model names get baked into a committed file.
  2. Verify `tests/test_mcp_pkg/test_mcp.py` still passes (command-level parity only, not param-freshness).
- `schema` excludes itself from its own output, so adding options to it (e.g. `--models-only`) never changes the bundle. It is also not exposed as an MCP tool — agents discover models via the `list_models` tool or the `wsinsight://models` resource, both of which read the registry through `wsinsight.modellib.models` (never `wsinfer_zoo.client.load_registry`, which ignores `WSINSIGHT_ZOO_REGISTRY_PATH` and reaches for HuggingFace).
- By default only STABLE commands are exposed: `run, patch, infer, ncomp, export, reg`. `--experimental` adds: `hplot, hplot-finalize, ecomp, tcomp, niche, niche-profile, agg, import`.
- Long-running commands (`run, patch, infer, ncomp, hplot, ecomp, tcomp, niche, agg`) return a `job_id`; clients poll `job_status`/`job_logs`/`cancel_job`. Poll `job_status` until it reports a terminal state (succeeded/failed/cancelled); on failure, fetch `job_logs` before retrying, and do not re-issue the same job while a prior one is still running.
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
