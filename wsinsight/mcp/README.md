# WSInsight MCP Server

A [Model Context Protocol](https://modelcontextprotocol.io/) server that
exposes the WSInsight CLI to MCP-compatible clients (Claude Desktop, the
VS Code Copilot MCP integration, custom agents, …) over **stdio** or
**Streamable HTTP**.

The server is built on [`fastmcp`](https://gofastmcp.com/) ≥ 2.0 and
auto-generates one MCP tool per stable WSInsight subcommand from the
single source of truth at
[`wsinsight/cli/cli_schema.json`](../cli/cli_schema.json).

---

## Install

```bash
pip install 'wsinsight[mcp]'
```

This adds `fastmcp` as a runtime dependency and registers the console
script `wsinsight-mcp`.

---

## Run

```bash
# stdio (default — what Claude Desktop / VS Code launch)
wsinsight-mcp

# Streamable HTTP, localhost-only
wsinsight-mcp --http 127.0.0.1:8765

# Cap concurrency (defaults to the number of visible GPUs)
wsinsight-mcp --max-concurrent 1

# Expose experimental tools (hplot, ecomp, tcomp, cme, hplot-finalize)
WSINSIGHT_EXPERIMENTAL=1 wsinsight-mcp --experimental
```

> **Security note.** The server is intended to run locally next to your
> data and GPUs. The HTTP transport binds whatever you tell it to; bind
> to `127.0.0.1` unless you have placed a reverse proxy with
> authentication in front of it.

---

## Tools

### Per-subcommand tools (auto-generated)

| Tool       | Long-running? | Maps to CLI                 |
|------------|---------------|-----------------------------|
| `run`      | yes           | `wsinsight run`             |
| `patch`    | yes           | `wsinsight patch`           |
| `infer`    | yes           | `wsinsight infer`           |
| `ncomp`    | yes           | `wsinsight ncomp`           |
| `export`   | no            | `wsinsight export`          |
| `reg`      | no            | `wsinsight reg`             |

Each tool's input schema mirrors the corresponding `--help` (parameter
names, types, defaults, descriptions) verbatim.

Long-running tools return immediately with a job descriptor:

```json
{
  "job_id": "01HZ…",
  "status": "started",
  "argv": ["infer", "--wsi-dir", "/data/tcga", ...],
  "hint": "Poll job_status(job_id='01HZ…') and job_logs(job_id='01HZ…')."
}
```

Short-running tools block (with a 600 s safety timeout) and return:

```json
{
  "status": "done",
  "returncode": 0,
  "argv": ["export", "--results-dir", "out/"],
  "duration_s": 3.27,
  "log_tail": ["…last 50 lines…"]
}
```

### Job-management meta-tools

* `job_status(job_id)` — snapshot of one job (status, pid, GPU, duration, return code, total log lines).
* `job_logs(job_id, since_line=0, max_lines=500)` — paginated stdout/stderr.
* `cancel_job(job_id)` — graceful cancel via `SIGINT`; a second call escalates to `SIGTERM`. The first signal triggers WSInsight's existing two-press cancellation handler so the current slide finishes cleanly.
* `list_jobs()` — all jobs known to this server (running and completed).

---

## Resources

* `wsinsight://schema` — the full CLI JSON schema.
* `wsinsight://models` — names of WSInfer/WSInsight model-zoo entries discovered locally.
* `wsinsight://results/{results_dir}/layout` — recursive listing (capped at 500 entries) of a results directory; the agent can confirm what was produced.

---

## Prompts

* `reproduce_tcga_crc` — step-by-step guide for reproducing the manuscript's TCGA-CRC analysis using the tools above.

---

## Client configuration

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or the equivalent on Windows / Linux:

```json
{
  "mcpServers": {
    "wsinsight": {
      "command": "wsinsight-mcp",
      "args": []
    }
  }
}
```

### VS Code (Copilot Chat MCP)

Add to your user `mcp.json`:

```json
{
  "servers": {
    "wsinsight": {
      "type": "stdio",
      "command": "wsinsight-mcp"
    }
  }
}
```

### Generic Streamable HTTP

```bash
wsinsight-mcp --http 127.0.0.1:8765
# then point any MCP HTTP client at http://127.0.0.1:8765/mcp
```

---

## Concurrency & GPU pinning

* Default `--max-concurrent` = number of visible GPUs (parsed from `CUDA_VISIBLE_DEVICES`, else `torch.cuda.device_count()`, else 1).
* Each running job is assigned exactly one GPU from the visible pool via `CUDA_VISIBLE_DEVICES` in the child process's environment.
* When the pool is full, `submit` blocks the caller (the MCP tool call) until a slot frees.

---

## Cancellation semantics

```
cancel_job(job_id)        → SIGINT  → wsinsight.cancel.request_cancel()
                                       finishes the current slide, then exits
cancel_job(job_id) again  → SIGTERM → hard stop
```

The first `SIGINT` is the same signal a user typing Ctrl-C in the
terminal would send, so it engages WSInsight's existing two-press
cancellation handler in [`wsinsight/cancel.py`](../cancel.py).

---

## Example session

```text
agent → list_jobs()                       → []
agent → infer(wsi_dir="/data/tcga",
              results_dir="/scratch/out",
              model_name="cellvit-pannuke-h-x40")
       → {job_id: "01HZ…", status: "started"}
agent → job_status(job_id="01HZ…")         → {status:"running", pid:1234, gpu_id:0, ...}
agent → job_logs(job_id="01HZ…")           → {lines:["…"], next_line:120}
agent → ncomp(wsi_dir="/data/tcga",
              results_dir="/scratch/out")  → another job_id
agent → export(results_dir="/scratch/out", geojson=true)
       → {status:"done", returncode:0, ...}
```
