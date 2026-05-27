#!/usr/bin/env bash
# 安装 pre-commit 钩子（防误删 + 完整性检查）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOK_SRC="${ROOT}/scripts/git_hooks/pre-commit"
HOOK_DST="${ROOT}/.git/hooks/pre-commit"

if [[ ! -d "${ROOT}/.git" ]]; then
  echo "[install_git_hooks] 无 .git，跳过" >&2
  exit 0
fi

mkdir -p "${ROOT}/.git/hooks"
cp "${HOOK_SRC}" "${HOOK_DST}"
chmod +x "${HOOK_DST}" "${ROOT}/scripts/verify_repo_integrity.sh"
echo "[install_git_hooks] 已安装 → .git/hooks/pre-commit"
