"""Run through Codespace SSH stdin to fast-forward a verified bundle safely.

Remote use: python - BUNDLE EXPECTED_COMMIT [--restart] < this_file.py
Only this task's identical, already-committed benchmark output may be moved to a
backup before checkout. Any other working-tree changes stop deployment.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def get_json(path):
    with urllib.request.urlopen("http://localhost:8000" + path, timeout=2) as response:
        assert response.status == 200
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("expected_commit")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    repository = Path("/workspaces/hotbox-edge-ai")
    os.chdir(repository)
    previous = git("rev-parse", "HEAD")
    git("bundle", "verify", args.bundle)
    git("fetch", args.bundle, "main")
    assert git("rev-parse", "FETCH_HEAD") == args.expected_commit
    subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD"], check=True)
    status = git("status", "--porcelain")
    owned = Path("artifacts/team_25/mission3/codespace_localhost")
    allowed = "?? " + str(owned) + "/"
    assert not status or status == allowed, "Preserve unexpected working-tree changes: " + status
    backup = None
    if status:
        for path in owned.rglob("*"):
            if path.is_file():
                committed = subprocess.check_output(["git", "show", "FETCH_HEAD:" + str(path)])
                assert path.read_bytes() == committed, "Remote experiment differs; do not overwrite: " + str(path)
        backup = tempfile.mkdtemp(prefix="hotbox-verified-benchmark-backup-")
        shutil.move(str(owned), str(Path(backup) / owned.name))
    if not args.restart:
        assert not git("diff", "--name-only", "HEAD", "FETCH_HEAD", "--", "app", "model"), "Application changed: restart required"
    git("merge", "--ff-only", "FETCH_HEAD")
    # The bundle was made from the local commit already verified on GitHub main.
    git("fetch", args.bundle, "main:refs/remotes/origin/main")
    assert git("rev-parse", "HEAD") == args.expected_commit
    assert not git("status", "--porcelain")
    pidfile = Path("/tmp/hotbox-team25-uvicorn.pid")
    if args.restart:
        if pidfile.exists():
            old_pid = int(pidfile.read_text().strip())
            proc = Path(f"/proc/{old_pid}")
            if proc.exists():
                command = (proc / "cmdline").read_bytes().split(b"\0")
                assert b"app.main:app" in command and b"8000" in command
                assert (proc / "cwd").resolve() == repository
                os.kill(old_pid, signal.SIGTERM)
                for _ in range(30):
                    if not proc.exists() or b"app.main:app" not in (proc / "cmdline").read_bytes().split(b"\0"):
                        break
                    time.sleep(0.1)
        with open("/tmp/hotbox-team25-uvicorn.log", "ab") as output:
            server = subprocess.Popen([sys.executable, "-u", "-m", "uvicorn", "app.main:app",
                                       "--host", "0.0.0.0", "--port", "8000"],
                                      stdin=subprocess.DEVNULL, stdout=output,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        pidfile.write_text(str(server.pid) + "\n")
        for attempt in range(30):
            assert server.poll() is None, "Uvicorn exited; inspect /tmp/hotbox-team25-uvicorn.log"
            try:
                health = get_json("/health")
                break
            except (OSError, ValueError):
                if attempt == 29:
                    raise
                time.sleep(0.5)
    health = get_json("/health")
    info = get_json("/model_info")
    assert health == {"status": "ok", "team": 25, "params": 385, "val_mae": 5.31}
    assert (info["total_params"], info["total_bytes_f32"], info["total_bytes_int8"], info["total_macs"]) == (385, 1540, 385, 352)
    reference = json.loads(Path("artifacts/team_25/mission2/verified_sha256.json").read_text())
    names = ["app/main.py", "app/model.py", "app/preprocessing.py", "model/model.pt", "model/config.json", "client/bench.py"]
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}
    for name in ("model/model.pt", "model/config.json"):
        assert hashes[name] == reference[name]
    print(json.dumps(dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          previous_commit=previous, deployed_commit=git("rev-parse", "HEAD"),
                          git_status=git("status", "--short", "--branch"),
                          preserved_benchmark_backup=backup, restarted=args.restart,
                          server_pid=int(pidfile.read_text()), sha256=hashes,
                          health=health, model_info=info), indent=2))


if __name__ == "__main__":
    main()
