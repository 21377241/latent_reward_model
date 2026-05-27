# 源码安全与防误删

## 原则

1. **所有 `.py` / `.sh` 改动先 commit，再跑长时训练**；远程 [21377241/latent_reward_model](https://github.com/21377241/latent_reward_model) 为备份源。
2. **禁止**在仓库根目录执行 `rm *.py`、`rm -rf scripts/` 等批量删除。
3. 长任务启动前运行完整性检查：
   ```bash
   bash scripts/verify_repo_integrity.sh
   ```
4. 安装 Git pre-commit（批量删除 >3 个 py/sh 会拦截）：
   ```bash
   bash scripts/install_git_hooks.sh
   ```

## 同步到 GitHub

```bash
export GITHUB_TOKEN="ghp_xxxx"   # 勿写入脚本或提交到 git
export GITHUB_OWNER="21377241"
bash scripts/push_to_github.sh
# 若 git push 失败：
export COMMIT_MSG="描述本次改动"
python scripts/sync_git_to_github_api.py
```

`sync_git_to_github_api.py` 默认从**磁盘**收集文件（`SYNC_FROM_GIT=1` 时才仅用 git HEAD），避免未 commit 的恢复代码漏推。

## 若源码再次丢失

1. 从 GitHub raw 或 `git clone https://github.com/21377241/latent_reward_model.git` 恢复。
2. 本地 `.git` 可用：
   ```bash
   GIT_CONFIG_GLOBAL=/tmp/latent_reward_model_gitconfig
   printf '[safe]\n\tdirectory = *\n' > "$GIT_CONFIG_GLOBAL"
   GIT_CONFIG_SYSTEM=/dev/null git -C /path/to/latent_reward_model checkout HEAD -- .
   ```

## 不在 Git 中的内容

`experiments/`、`data/`、`swanlog/` 由 `.gitignore` 排除；仅同步源码与配置。
