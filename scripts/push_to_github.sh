#!/usr/bin/env bash
# 在 GitHub 上创建仓库并将本地 latent_reward_model 与之同步。
#
# 用法:
#   export GITHUB_TOKEN="ghp_xxxx"          # repo 权限
#   export GITHUB_OWNER="your-username"     # 用户名或组织名
#   export GITHUB_REPO="latent_reward_model" # 可选，默认仓库名
#   export GITHUB_PRIVATE="false"           # true 则创建私有仓库
#   bash scripts/push_to_github.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# AFS/NFS 上 root 操作他人仓库时避免「dubious ownership」
export GIT_CONFIG_GLOBAL="${GIT_CONFIG_GLOBAL:-/tmp/latent_reward_model_gitconfig}"
if [[ ! -f "${GIT_CONFIG_GLOBAL}" ]]; then
  printf '[safe]\n\tdirectory = *\n' > "${GIT_CONFIG_GLOBAL}"
fi
export GIT_CONFIG_SYSTEM="${GIT_CONFIG_SYSTEM:-/dev/null}"

REPO_NAME="${GITHUB_REPO:-latent_reward_model}"
OWNER="${GITHUB_OWNER:-21377241}"
TOKEN="${GITHUB_TOKEN:-}"
PRIVATE="${GITHUB_PRIVATE:-false}"
BRANCH="${GITHUB_BRANCH:-main}"

if [[ -z "$OWNER" ]]; then
  echo "错误: 请设置 GITHUB_OWNER（GitHub 用户名或组织名）" >&2
  exit 1
fi
if [[ -z "$TOKEN" ]]; then
  echo "错误: 请设置 GITHUB_TOKEN（需 repo 权限的 Personal Access Token）" >&2
  exit 1
fi

if [[ -x scripts/verify_repo_integrity.sh ]]; then
  bash scripts/verify_repo_integrity.sh
fi

if [[ -x scripts/install_git_hooks.sh ]]; then
  bash scripts/install_git_hooks.sh
fi

if [[ ! -d .git ]]; then
  git init -b "$BRANCH"
fi

if ! git rev-parse HEAD >/dev/null 2>&1; then
  git add -A
  git -c user.name="${GIT_USER_NAME:-latent-reward-model}" \
      -c user.email="${GIT_USER_EMAIL:-latent-reward-model@users.noreply.github.com}" \
      commit -m "Initial commit: Latent Reward Model"
fi

API="https://api.github.com"
AUTH="Authorization: token ${TOKEN}"
ACCEPT="Accept: application/vnd.github+json"

# 若远程仓库已存在则跳过创建
if curl -sf -H "$AUTH" -H "$ACCEPT" "${API}/repos/${OWNER}/${REPO_NAME}" >/dev/null 2>&1; then
  echo "远程仓库 ${OWNER}/${REPO_NAME} 已存在，跳过创建。"
else
  echo "正在创建 GitHub 仓库 ${OWNER}/${REPO_NAME} ..."
  curl -sf -X POST -H "$AUTH" -H "$ACCEPT" "${API}/user/repos" \
    -d "{\"name\":\"${REPO_NAME}\",\"private\":${PRIVATE},\"description\":\"Latent multi-dimensional reward model\"}" \
    >/dev/null
  echo "仓库创建成功。"
fi

REMOTE_URL="https://${TOKEN}@github.com/${OWNER}/${REPO_NAME}.git"

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

if ! git push -u origin "$BRANCH"; then
  echo "[提示] git push 失败，改用 GitHub API 同步..." >&2
  export GITHUB_OWNER="${OWNER}"
  export GITHUB_REPO="${REPO_NAME}"
  export GITHUB_BRANCH="${BRANCH}"
  export COMMIT_MSG="${COMMIT_MSG:-Sync latent_reward_model $(date -Iseconds)}"
  python scripts/sync_git_to_github_api.py
  exit $?
fi
echo "已推送到 https://github.com/${OWNER}/${REPO_NAME}"
