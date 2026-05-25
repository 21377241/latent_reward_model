#!/usr/bin/env python3
"""
当 git push 无法连接 github.com:443 时，通过 api.github.com Git Data API 同步当前 HEAD。

用法:
  export GITHUB_TOKEN="ghp_xxxx"
  export GITHUB_OWNER="21377241"   # 可选，默认 21377241
  python scripts/sync_git_to_github_api.py
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = os.environ.get("GITHUB_OWNER", "21377241")
REPO = os.environ.get("GITHUB_REPO", "latent_reward_model")
BRANCH = os.environ.get("GITHUB_BRANCH", "main")
TOKEN = os.environ.get("GITHUB_TOKEN", "")
API = f"https://api.github.com/repos/{OWNER}/{REPO}"


def api(method: str, path: str, data: dict | None = None) -> dict:
    url = f"{API}{path}"
    body = None
    headers = {
        "Authorization": f"token {TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "latent-reward-model-sync",
    }
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {err}") from e


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True
    ).strip()


def list_files_at_head() -> list[tuple[str, str]]:
    """返回 (path, mode)。"""
    lines = git("ls-tree", "-r", "HEAD").splitlines()
    out = []
    for line in lines:
        meta, path = line.split("\t", 1)
        mode, typ, _sha = meta.split()
        if typ != "blob":
            continue
        out.append((path, mode))
    return out


def upload_blob(path: str) -> str:
    data = (ROOT / path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    resp = api("POST", "/git/blobs", {"content": b64, "encoding": "base64"})
    return resp["sha"]


def main() -> int:
    if not TOKEN:
        print("错误: 请设置 GITHUB_TOKEN（需 repo 权限）", file=sys.stderr)
        return 1

    commit_msg = git("log", "-1", "--format=%B")
    parent_sha = None
    try:
        ref = api("GET", f"/git/ref/heads/{BRANCH}")
        parent_sha = ref["object"]["sha"]
        print(f"远程 {BRANCH} 父提交: {parent_sha[:7]}")
    except RuntimeError as e:
        if "404" not in str(e):
            raise
        print(f"远程尚无 {BRANCH}，将创建首条提交")

    tree_entries = []
    for path, mode in list_files_at_head():
        blob_sha = upload_blob(path)
        tree_entries.append(
            {"path": path, "mode": mode, "type": "blob", "sha": blob_sha}
        )
        print(f"  blob {path}")

    tree = api("POST", "/git/trees", {"tree": tree_entries})
    commit_payload: dict = {"message": commit_msg, "tree": tree["sha"]}
    if parent_sha:
        commit_payload["parents"] = [parent_sha]

    commit = api("POST", "/git/commits", commit_payload)
    commit_sha = commit["sha"]
    print(f"新提交: {commit_sha[:7]} — {commit_msg.splitlines()[0]}")

    if parent_sha:
        api("PATCH", f"/git/refs/heads/{BRANCH}", {"sha": commit_sha, "force": False})
    else:
        api("POST", "/git/refs", {"ref": f"refs/heads/{BRANCH}", "sha": commit_sha})

    print(f"已更新 https://github.com/{OWNER}/{REPO}/tree/{BRANCH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
