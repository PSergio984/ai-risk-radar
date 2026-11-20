"""Execute notebook acquisition with external connections and vendor downloads forbidden."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def copy_notebook_workspace(destination: Path) -> Path:
    """Copy source only; never copy local data, releases, or model artifacts."""
    source = Path(__file__).resolve().parents[1]
    destination.mkdir(parents=True)
    shutil.copy(source / "config.json", destination / "config.json")
    shutil.copytree(source / "ai_risk_radar", destination / "ai_risk_radar")
    shutil.copytree(source / "notebooks", destination / "notebooks")
    return destination


def execute_notebook_offline(
    workspace: Path, mode: str | None = "fixture", *, full: bool = False,
) -> subprocess.CompletedProcess:
    """Run real notebook cells in a subprocess whose children inherit the network guard."""
    guard_dir = workspace / "offline_guard"
    guard_dir.mkdir(exist_ok=True)
    (guard_dir / "sitecustomize.py").write_text(
        '''import os
from pathlib import Path
import socket

def forbidden(reason):
    Path(os.environ["AI_RISK_RADAR_NETWORK_ATTEMPTS"]).write_text(reason)
    raise AssertionError("Offline notebook attempted external I/O: " + reason)

def local(address):
    return isinstance(address, tuple) and str(address[0]) in ("127.0.0.1", "::1", "localhost")

original_connect = socket.socket.connect
original_connect_ex = socket.socket.connect_ex
original_sendto = socket.socket.sendto
original_getaddrinfo = socket.getaddrinfo

def connect(self, address):
    if not local(address):
        forbidden("socket.connect " + str(address))
    return original_connect(self, address)

def connect_ex(self, address):
    if not local(address):
        forbidden("socket.connect_ex " + str(address))
    return original_connect_ex(self, address)

def sendto(self, data, *args):
    if args and not local(args[-1]):
        forbidden("socket.sendto " + str(args[-1]))
    return original_sendto(self, data, *args)

def getaddrinfo(host, *args, **kwargs):
    if host not in (None, "", "127.0.0.1", "::1", "localhost"):
        forbidden("DNS " + str(host))
    return original_getaddrinfo(host, *args, **kwargs)

socket.socket.connect = connect
socket.socket.connect_ex = connect_ex
socket.socket.sendto = sendto
socket.getaddrinfo = getaddrinfo

# yfinance can use native curl, which bypasses Python's socket methods.
import yfinance
def download(*args, **kwargs):
    forbidden("yfinance.download")
yfinance.download = download
''',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(guard_dir), str(workspace)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["AI_RISK_RADAR_NETWORK_ATTEMPTS"] = str(workspace / "network_attempt.txt")
    env.pop("AI_RISK_RADAR_NOTEBOOK_MODE", None)
    if mode is not None:
        env["AI_RISK_RADAR_NOTEBOOK_MODE"] = mode
    if full:
        command = [
            sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
            "--ExecutePreprocessor.kernel_name=python3",
            "--ExecutePreprocessor.timeout=120",
            str(workspace / "notebooks" / "end_to_end.ipynb"),
            "--output", "executed_end_to_end.ipynb",
        ]
    else:
        command = [sys.executable, "-c", '''import json
from pathlib import Path
notebook = json.loads(Path("notebooks/end_to_end.ipynb").read_text(encoding="utf-8"))
scope = {"__name__": "__main__"}
code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
for cell in code_cells[:2]:
    exec(compile("".join(cell["source"]), "end_to_end.ipynb", "exec"), scope)
''']
    return subprocess.run(
        command, cwd=workspace, env=env, capture_output=True, text=True,
        encoding="utf-8", timeout=180, check=False,
    )


def test_fixture_mode_never_downloads_and_preserves_synthetic_provenance(tmp_path: Path):
    """A fixture-mode regression to market acquisition must fail before any external I/O."""
    workspace = copy_notebook_workspace(tmp_path / "workspace")
    first = execute_notebook_offline(workspace)
    assert first.returncode == 0, first.stdout + first.stderr
    fixture_dir = workspace / "data" / "raw" / "notebook_fixture"
    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    assert manifest["is_synthetic"] is True
    assert manifest["source"] == "deterministic_offline_fixture"
    assert "SYNTHETIC DEMONSTRATION" in first.stdout
    assert "NOT MARKET EVIDENCE" in first.stdout
    second = execute_notebook_offline(workspace)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "SYNTHETIC DEMONSTRATION" in second.stdout
    assert json.loads((fixture_dir / "manifest.json").read_text())["hashes"] == manifest["hashes"]
    assert not (workspace / "network_attempt.txt").exists()
    assert not (workspace / "data" / "raw" / "snapshot").exists()
    assert not (workspace / "artifacts").exists()


def test_default_mode_requires_cache_without_attempting_download(tmp_path: Path):
    workspace = copy_notebook_workspace(tmp_path / "workspace")
    result = execute_notebook_offline(workspace, mode=None)
    assert result.returncode != 0
    assert "No cached notebook data" in result.stderr
    assert not (workspace / "network_attempt.txt").exists()
    assert not (workspace / "data").exists()


@pytest.mark.parametrize("mode", ["unknown", "FIXTURE-TYPO"])
def test_invalid_mode_fails_before_io(tmp_path: Path, mode: str):
    workspace = copy_notebook_workspace(tmp_path / "workspace")
    result = execute_notebook_offline(workspace, mode=mode)
    assert result.returncode != 0
    assert "Unsupported notebook mode" in result.stderr
    assert not (workspace / "network_attempt.txt").exists()
