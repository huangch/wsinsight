# qupath-extension-wsinsight

QuPath 0.7 extension that exposes the [WSInsight](https://github.com/huangch/wsinsight) CLI as a graphical tool. Each menu entry is a small form that collects arguments for one WSInsight subcommand and launches the `huangchtw/wsinsight:latest` Docker container with the right bind mounts, GPU flags, and environment variables. Output GeoJSON / OME-CSV files are optionally imported back into the active QuPath project on success.

## Requirements

- QuPath **0.7.0**.
- A working `docker` CLI on the host (Linux or macOS recommended).
- The NVIDIA Container Toolkit, if you want GPU acceleration.
- The WSInsight image pulled once:
  ```bash
  docker pull huangchtw/wsinsight:latest
  ```

## Build

```bash
cd qupath-extension-wsinsight
./gradlew clean shadowJar
```

The fat jar lands in `build/libs/qupath-extension-wsinsight-0.1.0-all.jar`. Drop it into QuPath's `extensions/` directory and restart QuPath.

## Configure

`Edit > Preferences > WSInsight` exposes:

| Preference | Purpose |
| --- | --- |
| Docker binary / image | `docker` executable and `huangchtw/wsinsight:latest` tag |
| GPUs | Value for `docker --gpus` (`all`, `none`, `device=0,1`, …) |
| Shared memory size | `docker --shm-size` (default `32g`) |
| Host WSI root (→ `/slides`) | Host directory bound to `/slides` |
| Host results root (→ `/results`) | Host directory bound to `/results` |
| Extra mounts | Additional `host:container` pairs, separated by commas, semicolons, or newlines |
| WSInfer Zoo registry path | Sets `WSINFER_ZOO_REGISTRY_PATH` inside the container |
| S3 storage options (JSON) | Sets `S3_STORAGE_OPTIONS` |
| Remote cache directory | Sets `WSINSIGHT_REMOTE_CACHE_DIR` |
| `KERAS_HOME` | Sets the Keras cache directory |
| Auto-import results | Load `*.geojson` and `*.ome.csv` back into the project on success |

## Run

`Extensions > WSInsight >` lists one entry per WSInsight CLI subcommand:

- **Run** — one-shot `patch → infer → hplot → ncomp → ecomp → tcomp → cme → export` pipeline
- **Patch**, **Infer**, **Region registration**
- **H-plot**, **H-plot finalize**
- **Neighborhood / Edge / Triad composition**
- **Cellular microenvironment**
- **Export GeoJSON / OME-CSV**

Each action opens a form pre-populated with sensible defaults. All path fields accept host paths; the extension rewrites them into container paths via the configured bind mounts before invoking `wsinsight`. Host paths that are not covered by any mount raise a clear error before the container is launched.

## Scope (v0.1)

- **Backend**: local Docker only.
- **Execution scope**: runs against whatever `--wsi-dir` the user picks. A project-wide batch mode is planned for v0.2.
- **OS**: tested on Linux and macOS. Windows is untested (`--user $(id -u)` is skipped).

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
