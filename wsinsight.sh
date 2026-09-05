#!/usr/bin/env bash
# wsinsight.sh - unified environment-aware runner for the wsinsight CLI.
#
# Manages BOTH backends (native + docker) without depending on the legacy
# wsinsight-docker-run.sh or the old env-prefix wsinsight.sh shim, which have
# both moved to bak_old_scripts/ and are kept only for reference.
#
# Subcommands:
#   run        -b {native,docker} -d DIR [--gpu ID|all] [--tmpdir DIR]
#               [--no-pull] [--dry-run] [WSINSIGHT_ARGS ...]
#   status                                                  # current effective config
#   doctor     [-b {native,docker}]                        # preflight
#   where                                                   # absolute path of this script
#   -h | --help
#   --version
#
#   `-d DIR` (a.k.a. `--data-dir`) is REQUIRED when -b docker is in effect: it
#   names the host directory that gets bind-mounted to /workspace inside the
#   container. WSINSIGHT_DATA_DIR supplies the same value non-interactively.
#
# Param-parsing rule (key design point):
#   Everything before the first wsinsight subcommand name (run, patch, infer, ...)
#   is consumed by THIS script (env control: -b, -d, --gpu, --tmpdir, --no-pull,
#   --dry-run).  Starting at (and including) the first wsinsight subcommand
#   name, every token is passed through to wsinsight verbatim. The wrapper
#   figures out where the boundary is by:
#     1. Absorbing its own -flag-with-arg / --flag / -h / --version tokens.
#     2. Recognizing status/doctor/where as script subcommands (NOT wsinsight's).
#     3. Honoring an explicit `--` delimiter (passthrough from there).
#     4. The first position arg is then scanned against a list of known wsinsight
#        subcommand names. If found, passthrough begins THERE.
#     5. Backward-scan fallback: if a position arg isn't a known subcommand,
#        scan the remainder of argv for the first known subcommand; if found,
#        shift to that position and passthrough from there. If nothing matches,
#        die with a list of known commands.
#
#   Unknown -X flags: in default (lenient) mode, warn and treat as wsinsight's;
#   in WSINSIGHT_STRICT=1 mode, die.
#
# Exit code 0 on success, 1 on user error (bad args, no wsinsight cmd found),
# 2 on infrastructure failure (docker daemon gone, wsinsight not on PATH and
# schema fetch failed).

set -euo pipefail

PROG="$(basename "$0")"
VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

IMAGE_ID="${WSINSIGHT_IMAGE:-huangchtw/wsinsight:latest}"
HF_CACHE_VOLUME="${WSINSIGHT_HF_CACHE_VOLUME:-wsinsight-hf-cache}"
COMMANDS_CACHE="$HOME/.cache/wsinsight/commands.txt"
COMMANDS_CACHE_TTL_SECONDS="${WSINSIGHT_COMMANDS_TTL_SECONDS:-86400}"

# `-d` / `--data-dir` is REQUIRED when backend=docker.  It mirrors clawpyter's
# `-d <notebook_dir>` and names the host directory that gets bind-mounted to
# /workspace inside the container.  WSINSIGHT_DATA_DIR is the matching env var
# (kept for CI / non-interactive use; the flag takes precedence when both set).
SCRIPT_DATA_DIR="${WSINSIGHT_DATA_DIR:-}"

# LAST-RESORT builtin subcommand list - updated as wsinsight evolves.
# Used only when neither the cache file nor a live `wsinsight schema --commands-only`
# invocation works. The bundled wsinsight/cli/cli_schema.json stays the source
# of truth; this list is a static fallback for offline / broken-shell scenarios.
_WS_BUILTIN_CMDS=(
    run patch infer reg ncomp export
    hplot hplot-finalize ecomp tcomp
    niche niche-profile agg import
)

# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------
print_usage() {
    cat <<EOF
$PROG $VERSION - run wsinsight with one of two backends (native | docker)

Usage:
  $PROG run     -b {native,docker} -d DIR | --data-dir DIR
                [--gpu ID|all] [--tmpdir DIR] [--no-pull] [--dry-run] [WSINSIGHT_ARGS ...]
  $PROG status
  $PROG doctor  [-b {native,docker}]
  $PROG where
  $PROG -h | --help
  $PROG --version

Backends:
  native    Invoke the wsinsight CLI on the host inside the activated env.
  docker    Run wsinsight inside the $IMAGE_ID container (auto-pull; persist HF cache volume).
            Requires -d DIR (the host dir bind-mounted to /workspace inside the
            container). WSINSIGHT_DATA_DIR provides the same value non-interactively.

Environment overrides:
  WSINSIGHT_BACKEND                Default backend when -b is not given (native | docker)
  WSINSIGHT_IMAGE                  Override the docker image tag
  WSINSIGHT_HF_CACHE_VOLUME        Override the persistent HF model cache volume name
  WSINSIGHT_COMMANDS_TTL_SECONDS   TTL for cached subcommand list (default 86400)
  WSINSIGHT_DATA_DIR                Default -d value when the flag is not given (used by docker)
  WSINSIGHT_STRICT=1               Die on unknown -X flags instead of warning + passthrough

Decision rule for argv parsing:
  Everything between the script's own flags and the first wsinsight subcommand
  name (run/patch/infer/...) is consumed by this script. From (and including)
  the first wsinsight subcommand name, every remaining argument is passed
  verbatim to wsinsight. Use -- to force passthrough explicitly.

Examples:
  $PROG run --wsi-dir slides/ --results-dir r/ --model X                              # native (default)
  $PROG -b docker -d /workspace/project run --wsi-dir slides/ --results-dir r/ --model X
  $PROG -b docker -d /workspace/project --gpu 0 --tmpdir /scratch --no-pull run --wsi-dir slides/
  $PROG -b docker -d /workspace/project --dry-run run --wsi-dir slides/    # print resolved docker command, do not run
  $PROG doctor
  $PROG where
EOF
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_log()  { printf '[%s] %s\n' "$PROG" "$*" >&2; }
_warn() { _log "WARNING: $*"; }
_die()  { _log "ERROR: $*"; exit "${2:-1}"; }

# ---------------------------------------------------------------------------
# wsinsight subcommand discovery
# ---------------------------------------------------------------------------
_get_wsinsight_command_list() {
    # Returns one command name per line on stdout.
    # 1. fresh cache wins
    # 2. live `wsinsight schema --commands-only` writes the cache
    # 3. builtin static fallback

    if [[ -f "$COMMANDS_CACHE" ]]; then
        local cache_age
        cache_age=$(( $(date +%s) - $(stat -c %Y "$COMMANDS_CACHE" 2>/dev/null || echo 0) ))
        if [[ $cache_age -lt $COMMANDS_CACHE_TTL_SECONDS ]]; then
            cat "$COMMANDS_CACHE"
            return 0
        fi
    fi

    if command -v wsinsight >/dev/null 2>&1; then
        local out
        if out="$(wsinsight schema --commands-only 2>/dev/null)" \
           && [[ -n "$out" ]] \
           && command -v python3 >/dev/null 2>&1; then
            local extracted
            if extracted="$(printf '%s' "$out" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    cmds = d.get('commands') or []
    if isinstance(cmds, dict): cmds = list(cmds.keys())
    for c in cmds: print(c)
except Exception:
    sys.exit(1)
")" && [[ -n "$extracted" ]]; then
                mkdir -p "$(dirname "$COMMANDS_CACHE")"
                printf '%s\ns\n' "$extracted" > "$COMMANDS_CACHE" 2>/dev/null || true
                # The above can mis-escape; just use cat-from-variable:
                printf '%s\n' "$extracted" > "$COMMANDS_CACHE"
                printf '%s\n' "$extracted"
                return 0
            fi
        fi
    fi

    # LAST RESORT
    _warn "could not refresh subcommand cache; using builtin list (last known good)"
    printf '%s\n' "${_WS_BUILTIN_CMDS[@]}"
}

_is_wsinsight_cmd() {
    local needle="$1" cmd
    for cmd in "${WS_CMDS[@]}"; do
        [[ "$cmd" == "$needle" ]] && return 0
    done
    return 1
}

# Load subcommand list (once per invocation)
WS_CMDS=()
while IFS= read -r c; do
    [[ -n "$c" ]] && WS_CMDS+=("$c")
done < <(_get_wsinsight_command_list)
[[ ${#WS_CMDS[@]} -gt 0 ]] || _die "could not determine wsinsight subcommand list (cache + fallback both empty)"

# ---------------------------------------------------------------------------
# Phase 1: parse argv (does NOT depend on backend choice)
# ---------------------------------------------------------------------------
SCRIPT_BACKEND=""
SCRIPT_GPU=""
SCRIPT_TMPDIR=""
SCRIPT_NO_PULL=0
SCRIPT_DRY_RUN=0
EXTRA_CMD=""     # status | doctor | where | ""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--backend)
            [[ -n "${2:-}" ]] || _die "-b / --backend requires a value"
            SCRIPT_BACKEND="$2"; shift 2 ;;
        -b=*|--backend=*)
            SCRIPT_BACKEND="${1#*=}"; shift ;;
        --gpu)
            [[ -n "${2:-}" ]] || _die "--gpu requires a value"
            SCRIPT_GPU="$2"; shift 2 ;;
        --gpu=*)
            SCRIPT_GPU="${1#*=}"; shift ;;
        --tmpdir)
            [[ -n "${2:-}" ]] || _die "--tmpdir requires a value"
            SCRIPT_TMPDIR="$2"; shift 2 ;;
        --tmpdir=*)
            SCRIPT_TMPDIR="${1#*=}"; shift ;;
        -d|--data-dir)
            [[ -n "${2:-}" ]] || _die "-d / --data-dir requires a value"
            SCRIPT_DATA_DIR="$2"; shift 2 ;;
        -d=*|--data-dir=*)
            SCRIPT_DATA_DIR="${1#*=}"; shift ;;
        --no-pull)
            SCRIPT_NO_PULL=1; shift ;;
        --dry-run|--dryrun)
            SCRIPT_DRY_RUN=1; shift ;;
        -h|--help)
            print_usage; exit 0 ;;
        --version)
            echo "$PROG $VERSION"; exit 0 ;;
        --)
            shift
            break ;;                       # explicit delimiter
        status|doctor|where)
            EXTRA_CMD="$1"; shift
            break ;;                       # script's own subcommands (priority)
        -*)
            if [[ "${WSINSIGHT_STRICT:-0}" == "1" ]]; then
                _die "unknown option: $1 (WSINSIGHT_STRICT=1; known wsinsight subcommands: ${WS_CMDS[*]})"
            else
                _warn "unrecognized script flag: $1 -- treating as wsinsight's (use WSINSIGHT_STRICT=1 to enforce)"
                break                       # let wsinsight decide
            fi
            ;;
        *)
            # Position arg. Is it a known wsinsight subcommand?
            if _is_wsinsight_cmd "$1"; then
                break                       # passthrough starts here
            fi
            # Not a known cmd. Backward-scan the remainder for one.
            matched=0; found_idx=0; i=1
            for arg in "$@"; do
                if _is_wsinsight_cmd "$arg"; then
                    matched=1; found_idx=$i; break
                fi
                i=$((i + 1))
            done
            if [[ $matched -eq 1 ]]; then
                shift $((found_idx - 1))
                break
            fi
            _die "no wsinsight subcommand found in: $* (known: ${WS_CMDS[*]})"
            ;;
    esac
done
WSINSIGHT_ARGS=("$@")

# ---------------------------------------------------------------------------
# Phase 2: dispatch on EXTRA_CMD (backend choice happens HERE)
# ---------------------------------------------------------------------------

if [[ -z "$SCRIPT_BACKEND" ]]; then
    SCRIPT_BACKEND="${WSINSIGHT_BACKEND:-native}"
fi
case "$SCRIPT_BACKEND" in
    native|docker) ;;
    *) _die "unknown backend: '$SCRIPT_BACKEND' (use 'native' or 'docker')" ;;
esac

# Map --gpu X to docker --gpus X with sane defaults. The result is APPENDED
# to the array named by $1 so that `--gpus all` stays as two separate argv
# tokens (`--gpus`, `all`), which docker requires; a single quoted string
# `--gpus all` is rejected as an unknown flag.
#
# Caller usage:
#   local -a gpus=()
#   resolve_docker_gpus_flag gpus
#   docker "${gpus[@]}" ...
#
# For the dry-run path (we just want to print the resolved command line), we
# append shell-escaped tokens; `printf %q` re-quotes elements with embedded
# spaces, which when printed looks like `--gpus\ all`, matching what bash
# would see if the user typed it. (`set -x` style.)
resolve_docker_gpus_flag() {
    local -n _out="$1"
    case "$SCRIPT_GPU" in
        all|"")                 _out+=(--gpus all) ;;
        device=*|"capabilities"=*)
                                  _out+=(--gpus "$SCRIPT_GPU") ;;
        *)                       _out+=(--gpus "device=$SCRIPT_GPU") ;;
    esac
}

cmd_status() {
    cat <<EOF
PROG         : $PROG ($SCRIPT_DIR/$PROG)
VERSION      : $VERSION
Backend      : $SCRIPT_BACKEND$( [[ $SCRIPT_DRY_RUN -eq 1 ]] && echo " (dry-run)" )
Data dir(-d) : ${SCRIPT_DATA_DIR:-<unset - required only for -b docker>}
GPU          : ${SCRIPT_GPU:-<default>$( [[ $SCRIPT_BACKEND == docker ]] && echo " (all)" )}
Tmpdir       : ${SCRIPT_TMPDIR:-<unchanged>}
No-pull      : $SCRIPT_NO_PULL
Image        : $IMAGE_ID
HF cache vol : $HF_CACHE_VOLUME
wsinsight    : $(command -v wsinsight 2>/dev/null || echo "<not on PATH>")
docker       : $(command -v docker 2>/dev/null || echo "<not on PATH>")
Subcommands  : ${#WS_CMDS[@]} known; cache: $COMMANDS_CACHE
Pass-through : ${#WSINSIGHT_ARGS[@]} arg(s)${WSINSIGHT_ARGS[*]:+: ${WSINSIGHT_ARGS[*]}}
EOF
}

cmd_doctor() {
    local target="${DOC_BACKEND:-$SCRIPT_BACKEND}"
    local rc=0
    printf 'doctor (backend=%s)\n' "$target"
    case "$target" in
        native)
            local wsi
            if wsi="$(command -v wsinsight)"; then
                printf '  [OK]  wsinsight on PATH: %s\n' "$wsi"
                if "$wsi" --version >/dev/null 2>&1; then
                    printf '  [OK]  wsinsight --version runs\n'
                else
                    printf '  [WARN] wsinsight --version failed (env may be incomplete)\n'
                fi
            else
                printf '  [FAIL] wsinsight not on PATH (activate the wsi conda env, or use --backend docker)\n'
                rc=2
            fi
            if [[ -n "${WSINSIGHT_ZOO_REGISTRY_PATH:-}" && -f "${WSINSIGHT_ZOO_REGISTRY_PATH}" ]]; then
                printf '  [OK]  WSINSIGHT_ZOO_REGISTRY_PATH: %s\n' "$WSINSIGHT_ZOO_REGISTRY_PATH"
            else
                printf '  [INFO] WSINSIGHT_ZOO_REGISTRY_PATH not set (will use default)\n'
            fi
            if command -v nvidia-smi >/dev/null 2>&1; then
                local gpu_count
                gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
                printf '  [OK]  nvidia-smi reports %s GPU(s)\n' "$gpu_count"
            else
                printf '  [INFO] nvidia-smi not on PATH (CPU-only inference still works)\n'
            fi
            ;;
        docker)
            if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
                printf '  [OK]  docker daemon reachable\n'
                if docker image inspect "$IMAGE_ID" >/dev/null 2>&1; then
                    printf '  [OK]  image present locally: %s\n' "$IMAGE_ID"
                else
                    printf '  [INFO] image not local; will pull on first run: %s\n' "$IMAGE_ID"
                fi
                if docker volume inspect "$HF_CACHE_VOLUME" >/dev/null 2>&1; then
                    printf '  [OK]  HF cache volume exists: %s\n' "$HF_CACHE_VOLUME"
                else
                    printf '  [INFO] HF cache volume does not exist; will be auto-created: %s\n' "$HF_CACHE_VOLUME"
                fi
                if command -v nvidia-smi >/dev/null 2>&1; then
                    local gpu_count
                    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
                    printf '  [OK]  nvidia-smi reports %s GPU(s)\n' "$gpu_count"
                else
                    printf '  [WARN] nvidia-smi missing; GPU passthrough will not work\n'
                    rc=1
                fi
            else
                printf '  [FAIL] docker not reachable\n'
                rc=2
            fi
            ;;
    esac
    return $rc
}

cmd_where() {
    echo "$SCRIPT_DIR/$PROG"
}

# Build the resolved docker run args (printed by --dry-run). Takes the host
# data dir as $1 so this function doesn't depend on main()'s scope.
build_docker_command() {
    local data_dir="$1"
    local -a parts=(docker run --rm -it)
    resolve_docker_gpus_flag parts
    parts+=(--shm-size=32g --init)
    [[ -n "${HOST_UID:-}"   ]] && parts+=(-e HOST_UID)
    [[ -n "${HOST_GID:-}"   ]] && parts+=(-e HOST_GID)
    [[ -n "$SCRIPT_TMPDIR"  ]] && parts+=(-e TMPDIR="$SCRIPT_TMPDIR")
    parts+=(-v "$data_dir":/workspace -v "$HF_CACHE_VOLUME":/app/hf-cache)
    parts+=("$IMAGE_ID")
    if [[ ${#WSINSIGHT_ARGS[@]} -gt 0 ]]; then
        parts+=(wsinsight "${WSINSIGHT_ARGS[@]}")
    fi
    # Shell-quote each token with printf %q. Tokens with embedded spaces
    # (e.g. `--gpus all` -> `--gpus` and `all`) end up as separate quoted
    # tokens, which is exactly what bash will see when re-parsing. The
    # trailing space is dropped naturally because printf doesn't add it
    # after the last argument.
    local i
    for i in "${!parts[@]}"; do
        printf '%q' "${parts[$i]}"
        if [[ $i -lt $((${#parts[@]} - 1)) ]]; then printf ' '; fi
    done
}

case "$EXTRA_CMD" in
    where)   cmd_where; exit 0 ;;
    status)  cmd_status; exit 0 ;;
    doctor)  cmd_doctor; exit $? ;;
    "")      : ;;
    *)       _die "unknown subcommand: $EXTRA_CMD" ;;
esac

main() {
    if [[ ${#WSINSIGHT_ARGS[@]} -eq 0 && -z "$EXTRA_CMD" ]]; then
        print_usage
        exit 0
    fi
    if [[ ${#WSINSIGHT_ARGS[@]} -eq 0 ]]; then
        if [[ "$SCRIPT_BACKEND" == "native" ]]; then
            _die "no wsinsight subcommand supplied. Try: $PROG run --wsi-dir ... --results-dir ... --model X"
        fi
        # docker with no args -> interactive shell. Allowed by convention.
    fi

    case "$SCRIPT_BACKEND" in
        native)
            if [[ $SCRIPT_DRY_RUN -eq 1 ]]; then
                local wsi_bin_dry
                if [[ -n "${WSINSIGHT_BIN:-}" ]]; then
                    wsi_bin_dry="$WSINSIGHT_BIN"
                elif command -v wsinsight >/dev/null 2>&1; then
                    wsi_bin_dry="$(command -v wsinsight)"
                else
                    wsi_bin_dry="/opt/anaconda3/envs/wsi/bin/wsinsight"
                fi
                printf '+ env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV PATH=/usr/bin:/bin '
                printf 'WSINSIGHT_EXPERIMENTAL=%q ' "${WSINSIGHT_EXPERIMENTAL:-1}"
                printf 'WSINSIGHT_ZOO_REGISTRY_PATH=%q ' "${WSINSIGHT_ZOO_REGISTRY_PATH:-}"
                printf 'KERAS_HOME=%q HF_HOME=%q HF_HUB_ENABLE_HF_TRANSFER=%q ' \
                    "${KERAS_HOME:-}" "${HF_HOME:-}" "${HF_HUB_ENABLE_HF_TRANSFER:-1}"
                printf 'SSL_CERT_FILE=%q %q %q\n' \
                    "${SSL_CERT_FILE:-/etc/pki/tls/certs/ca-bundle.crt}" \
                    "$wsi_bin_dry" "${WSINSIGHT_ARGS[*]}"
                exit 0
            fi
            local wsi_bin
            if [[ -n "${WSINSIGHT_BIN:-}" ]]; then
                wsi_bin="$WSINSIGHT_BIN"
            elif command -v wsinsight >/dev/null 2>&1; then
                wsi_bin="$(command -v wsinsight)"
            else
                wsi_bin="/opt/anaconda3/envs/wsi/bin/wsinsight"
            fi
            if [[ ! -x "$wsi_bin" ]]; then
                _die "wsinsight interpreter not found at '$wsi_bin'. Activate the wsi conda env or set WSINSIGHT_BIN."
            fi
            exec env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
                PATH=/usr/bin:/bin \
                WSINSIGHT_EXPERIMENTAL="${WSINSIGHT_EXPERIMENTAL:-1}" \
                WSINSIGHT_ZOO_REGISTRY_PATH="${WSINSIGHT_ZOO_REGISTRY_PATH:-}" \
                KERAS_HOME="${KERAS_HOME:-}" \
                HF_HOME="${HF_HOME:-}" \
                HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" \
                SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/pki/tls/certs/ca-bundle.crt}" \
                "$wsi_bin" "${WSINSIGHT_ARGS[@]}"
            ;;
        docker)
            # Determine data dir: required for docker. The flag (-d / --data-dir)
            # takes precedence; WSINSIGHT_DATA_DIR is the env-var form (CI / non-interactive).
            local DATA_DIR="$SCRIPT_DATA_DIR"
            if [[ -z "$DATA_DIR" ]]; then
                _die "docker backend needs a data dir. Pass -d DIR (or --data-dir DIR), or set WSINSIGHT_DATA_DIR=/path. The host dir is bind-mounted to /workspace inside the container; WSINSIGHT_ARGS sees /workspace as cwd."
            fi
            if [[ ! -d "$DATA_DIR" ]]; then
                _die "--data-dir '$DATA_DIR' does not exist (or is not a directory). Pass -d DIR pointing at an existing path."
            fi
            if [[ $SCRIPT_DRY_RUN -eq 1 ]]; then
                echo "+ $(build_docker_command "$DATA_DIR")"
                exit 0
            fi
            if [[ $SCRIPT_NO_PULL -eq 0 ]]; then
                docker pull "$IMAGE_ID" >/dev/null 2>&1 \
                    || _warn "docker pull failed; using local image (if any)"
            fi
            local -a docker_args=( run --rm -it )
            resolve_docker_gpus_flag docker_args
            docker_args+=( --shm-size=32g --init )
            [[ -n "${HOST_UID:-}"  ]] && docker_args+=( -e HOST_UID )
            [[ -n "${HOST_GID:-}"  ]] && docker_args+=( -e HOST_GID )
            [[ -n "$SCRIPT_TMPDIR" ]] && docker_args+=( -e TMPDIR="$SCRIPT_TMPDIR" )
            docker_args+=( -v "$DATA_DIR":/workspace -v "$HF_CACHE_VOLUME":/app/hf-cache )
            docker_args+=( "$IMAGE_ID" )
            if [[ ${#WSINSIGHT_ARGS[@]} -gt 0 ]]; then
                docker_args+=( wsinsight "${WSINSIGHT_ARGS[@]}" )
            fi
            exec docker "${docker_args[@]}"
            ;;
    esac
}

main "$@"
