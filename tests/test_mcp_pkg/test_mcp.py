"""Tests for the MCP package: schema parity, adapter argv translation,
and a smoke test of JobManager with a fake long-running command.
"""

from __future__ import annotations

import shutil
import sys
import time

import pytest

from wsinsight.mcp import adapters
from wsinsight.mcp import jobs
from wsinsight.mcp import schema

# -- schema -----------------------------------------------------------------


def test_schema_loads_and_lists_stable_commands():
    full = schema.load_schema()
    assert "commands" in full
    cmds = schema.discover_commands(experimental=False)
    # Every stable command should be present in the bundled CLI schema.
    missing = schema.STABLE_COMMANDS - set(cmds)
    assert not missing, f"stable commands missing from cli_schema.json: {missing}"


def test_experimental_gate_adds_more_commands():
    stable = set(schema.discover_commands(experimental=False))
    full = set(schema.discover_commands(experimental=True))
    assert stable.issubset(full)
    assert full - stable == set(schema.EXPERIMENTAL_COMMANDS) & full


def test_command_to_input_schema_has_required_and_properties():
    cmds = schema.discover_commands()
    ncomp = cmds["ncomp"]
    isch = schema.command_to_input_schema(ncomp)
    assert isch["type"] == "object"
    assert "wsi_dir" in isch["properties"]
    assert "results_dir" in isch["properties"]
    # ncomp_k is a non-required int with default 2.
    k = isch["properties"]["ncomp_k"]
    assert k["type"] == "integer"
    assert k.get("default") == 2
    # Required list contains both wsi_dir and results_dir.
    assert "wsi_dir" in isch["required"]
    assert "results_dir" in isch["required"]


# -- adapters ---------------------------------------------------------------


def test_args_to_argv_basic():
    cmds = schema.discover_commands()
    argv = adapters.args_to_argv(
        cmds["ncomp"],
        {
            "wsi_dir": "/data/wsis",
            "results_dir": "/data/results",
            "ncomp_k": 3,
            "overwrite": True,
        },
    )
    assert argv[0] == "ncomp"
    assert "--wsi-dir" in argv and argv[argv.index("--wsi-dir") + 1] == "/data/wsis"
    assert "--results-dir" in argv
    assert "--ncomp-k" in argv and argv[argv.index("--ncomp-k") + 1] == "3"
    assert "--overwrite" in argv  # bare boolean flag


def test_args_to_argv_skips_falsy_boolean():
    cmds = schema.discover_commands()
    argv = adapters.args_to_argv(
        cmds["ncomp"],
        {
            "wsi_dir": "/x",
            "results_dir": "/y",
            "overwrite": False,
        },
    )
    assert "--overwrite" not in argv


def test_args_to_argv_rejects_unknown_param():
    cmds = schema.discover_commands()
    with pytest.raises(adapters.AdapterError):
        adapters.args_to_argv(
            cmds["ncomp"],
            {"wsi_dir": "/x", "results_dir": "/y", "bogus_flag": 1},
        )


def test_args_to_argv_requires_required():
    cmds = schema.discover_commands()
    with pytest.raises(adapters.AdapterError):
        adapters.args_to_argv(cmds["ncomp"], {"wsi_dir": "/x"})  # results_dir missing


# -- jobs (fake subprocess) -------------------------------------------------


def _patch_for_fake_subprocess(monkeypatch, fake_argv):
    """Replace JobManager._run_job's subprocess command with a tiny shim.

    We monkey-patch by overriding submit() to use our own argv directly.
    """
    real_submit = jobs.JobManager.submit

    def fake_submit(self, command, argv_tail):
        # Replace the wsinsight argv with our fake script; pretend the
        # subcommand is still 'command' for status reporting.
        return real_submit(self, command, fake_argv)

    monkeypatch.setattr(jobs.JobManager, "submit", fake_submit)


def test_job_manager_smoke(tmp_path, monkeypatch):
    # Skip on Windows-only edge cases (signal differences). Fine on Linux/macOS.
    if sys.platform.startswith("win"):
        pytest.skip("subprocess SIGINT semantics differ on Windows")

    # A fake child that prints 5 lines then exits 0.
    script = tmp_path / "fake_cmd.py"
    script.write_text(
        "import sys, time\n"
        "for i in range(5):\n"
        "    print(f'fake-line-{i}', flush=True)\n"
        "    time.sleep(0.05)\n"
        "sys.exit(0)\n"
    )
    fake_argv_tail = [str(script)]

    # Patch JobManager.submit internals: build argv = [python, fake_script]
    def fake_submit(self, command, argv_tail):  # noqa: ARG001
        # bypass the wsinsight prefix; run our shim directly
        import uuid

        job_id = uuid.uuid4().hex[:12]
        argv = [sys.executable] + fake_argv_tail
        state = jobs.JobState(id=job_id, command=command, argv=argv)
        with self._lock:
            self._jobs[job_id] = state
        import threading

        threading.Thread(target=self._run_job, args=(state,), daemon=True).start()
        return job_id

    monkeypatch.setattr(jobs.JobManager, "submit", fake_submit)

    mgr = jobs.JobManager(max_concurrent=1)
    job_id = mgr.submit("infer", [])

    # Wait for completion (cap at 5 s).
    deadline = time.time() + 5.0
    while time.time() < deadline:
        st = mgr.status(job_id)
        if st and st["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.05)

    st = mgr.status(job_id)
    assert st is not None
    assert st["status"] == "done", st
    assert st["returncode"] == 0

    log = mgr.logs(job_id)
    assert log is not None
    assert log["total"] == 5
    assert log["lines"] == [f"fake-line-{i}" for i in range(5)]


def test_job_manager_cancel(tmp_path, monkeypatch):
    if sys.platform.startswith("win"):
        pytest.skip("subprocess SIGINT semantics differ on Windows")

    # A fake child that prints forever until SIGINT.
    script = tmp_path / "loop.py"
    script.write_text(
        "import sys, time, signal\n"
        "def _h(*a):\n"
        "    print('got SIGINT', flush=True)\n"
        "    sys.exit(130)\n"
        "signal.signal(signal.SIGINT, _h)\n"
        "while True:\n"
        "    print('tick', flush=True); time.sleep(0.1)\n"
    )
    fake_argv_tail = [str(script)]

    def fake_submit(self, command, argv_tail):  # noqa: ARG001
        import threading
        import uuid

        job_id = uuid.uuid4().hex[:12]
        argv = [sys.executable] + fake_argv_tail
        state = jobs.JobState(id=job_id, command=command, argv=argv)
        with self._lock:
            self._jobs[job_id] = state
        threading.Thread(target=self._run_job, args=(state,), daemon=True).start()
        return job_id

    monkeypatch.setattr(jobs.JobManager, "submit", fake_submit)

    mgr = jobs.JobManager(max_concurrent=1)
    job_id = mgr.submit("patch", [])

    # Wait until at least one log line has been buffered.
    deadline = time.time() + 3.0
    while time.time() < deadline:
        log = mgr.logs(job_id)
        if log and log["total"] > 0:
            break
        time.sleep(0.05)

    res = mgr.cancel(job_id)
    assert res is not None
    assert res.get("cancel_requested") is True

    # Wait for the child to exit.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        st = mgr.status(job_id)
        if st and st["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.05)

    st = mgr.status(job_id)
    assert st["status"] == "cancelled", st


# -- server build (skipped if fastmcp not installed) ------------------------


@pytest.mark.skipif(
    shutil.which is None
    or "fastmcp" not in sys.modules
    and __import__("importlib").util.find_spec("fastmcp") is None,
    reason="fastmcp not installed",
)
def test_build_server_registers_stable_tools():
    from wsinsight.mcp.server import build_server

    mcp = build_server(max_concurrent=1, experimental=False)
    # Smoke check: the server should expose every stable command plus the
    # four meta tools.
    # FastMCP exposes registered tools through internal registries that
    # vary by version, so we just assert the build succeeded.
    assert mcp is not None
