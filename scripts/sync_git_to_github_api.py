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


SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "experiments",
    "data",
    "swanlog",
    "wandb",
    "venv",
    "env",
    ".conda",
    "output",
    "logs",
    "runs",
}
SKIP_FILE_NAMES = {"LatentMRM_code.zip"}


def git_env() -> dict[str, str]:
    env = os.environ.copy()
    cfg = Path("/tmp/latent_reward_model_gitconfig")
    cfg.write_text(
        "[safe]\n\tdirectory = *\n",
        encoding="utf-8",
    )
    env["GIT_CONFIG_GLOBAL"] = str(cfg)
    env["GIT_CONFIG_SYSTEM"] = "/dev/null"
    return env


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args],
        text=True,
        env=git_env(),
    ).strip()


def collect_files_from_disk() -> list[tuple[str, str]]:
    """不依赖 git 可用性，按 .gitignore 规则遍历仓库文件。"""
    out: list[tuple[str, str]] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        parts = rel.split("/")
        if any(p in SKIP_DIR_NAMES for p in parts):
            continue
        if path.name in SKIP_FILE_NAMES:
            continue
        if path.suffix in (".pyc", ".pyo"):
            continue
        if path.suffix in (".pth", ".pt", ".ckpt", ".safetensors", ".bin"):
            continue
        if path.suffix in (".jsonl", ".csv"):
            continue
        if path.suffix == ".json" and path.name != "ds_zero2.json":
            continue
        if path.suffix == ".json" and "tokenizer" in path.name:
            continue
        mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        out.append((rel, mode))
    return out


def list_files_at_head() -> list[tuple[str, str]]:
    """返回 (path, mode)；优先 git HEAD，失败则扫盘。"""
    try:
        lines = git("ls-tree", "-r", "HEAD").splitlines()
        out = []
        for line in lines:
            meta, path = line.split("\t", 1)
            mode, typ, _sha = meta.split()
            if typ != "blob":
                continue
            out.append((path, mode))
        return out
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[warn] git 不可用 ({e})，改从磁盘收集文件", file=sys.stderr)
        return collect_files_from_disk()


def upload_blob(path: str) -> str:
    data = (ROOT / path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    resp = api("POST", "/git/blobs", {"content": b64, "encoding": "base64"})
    return resp["sha"]


def main() -> int:
    if not TOKEN:
        print("错误: 请设置 GITHUB_TOKEN（需 repo 权限）", file=sys.stderr)
        return 1

    commit_msg = os.environ.get("COMMIT_MSG", "").strip()
    if not commit_msg:
        try:
            commit_msg = git("log", "-1", "--format=%B")
        except (subprocess.CalledProcessError, FileNotFoundError):
            msg_file = ROOT / ".git" / "COMMIT_EDITMSG"
            commit_msg = (
                msg_file.read_text(encoding="utf-8").strip()
                if msg_file.is_file()
                else "Sync latent_reward_model"
            )
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
