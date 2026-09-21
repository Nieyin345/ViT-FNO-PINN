#!/bin/bash
# =====================================================================
# 代码上传（本地 -> 服务器），增量，基于 git。
#
# 用法:
#   bash deploy/sync_code.sh --init     # 首次：建裸仓库、push、检出
#   bash deploy/sync_code.sh            # 之后：增量 push + 更新工作区
#   bash deploy/sync_code.sh --status   # 看两边 HEAD 与本地未提交改动
#
# 结构：服务器上放一个**裸仓库**当中间站，工作区从它检出：
#
#     本地 repo  --git push-->  $REMOTE_BARE  --checkout-->  $REMOTE_REPO
#
# 好处：工作区永远是干净检出（本地怎么折腾都不影响），且第二次起只传增量。
# 代码走 git 而非 rsync，是因为本地 Git Bash **没有 rsync**（实测），
# 而 git 本来就该用来管代码。
#
# 放 /opt：本脚本用于**配置阶段**（镜像前）。镜像会把 /opt 打进去，
# 这样镜像里就带着代码。镜像之后若只改代码，重推即可。
# =====================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$HERE/config.sh"

BRANCH="$(git -C "$LOCAL_ROOT" rev-parse --abbrev-ref HEAD)"
MODE="${1:-push}"

hr() { echo "----------------------------------------------------------------"; }

ensure_dirs() {
    $SSH "sudo mkdir -p '$OPT_ROOT' '$REMOTE_BARE' '$REMOTE_REPO' && \
          sudo chown -R \$(id -u):\$(id -g) '$OPT_ROOT'" || return 1
}

init_remote() {
    hr; echo "初始化服务器裸仓库"; hr
    ensure_dirs || { echo "!!! 建目录失败"; return 1; }
    $SSH "git -C '$REMOTE_BARE' rev-parse --is-bare-repository 2>/dev/null | grep -q true || \
          git init --bare -q '$REMOTE_BARE'; \
          git -C '$REMOTE_BARE' config receive.denyCurrentBranch ignore" || return 1
    echo "裸仓库就绪: $REMOTE_BARE"
}

push_code() {
    hr; echo "推送 '$BRANCH' -> $NODE"; hr
    # 用裸仓库的绝对路径（ssh:// 形式需要端口/路径都规整，这里更省事）
    git -C "$LOCAL_ROOT" push -q "ssh://$REMOTE_USER@$NODE$REMOTE_BARE" "$BRANCH" || {
        echo "!!! push 失败（远程目录还没建？先跑 --init）"; return 1; }
    echo "push 完成"

    # 工作区：首次 clone，之后 fetch + 强制对齐
    $SSH "if [ -d '$REMOTE_REPO/.git' ]; then \
              cd '$REMOTE_REPO' && git fetch -q origin '$BRANCH' && \
              git checkout -q '$BRANCH' 2>/dev/null || git checkout -q -b '$BRANCH' FETCH_HEAD; \
              git reset -q --hard FETCH_HEAD; \
          else \
              git clone -q -b '$BRANCH' '$REMOTE_BARE' '$REMOTE_REPO'; \
          fi; \
          cd '$REMOTE_REPO' && echo '服务器工作区:' && \
          git log -1 --format='  %h %s (%cr)' && \
          echo -n '  分支: ' && git rev-parse --abbrev-ref HEAD" || {
        echo "!!! 服务器检出失败"; return 1; }
}

status() {
    hr; echo "本地 HEAD"; hr
    git -C "$LOCAL_ROOT" log -1 --format='  %h %s'
    hr; echo "服务器 HEAD"; hr
    $SSH "cd '$REMOTE_REPO' 2>/dev/null && git log -1 --format='  %h %s' || echo '  (服务器还没有工作区)'"
    hr; echo "本地未提交改动"; hr
    git -C "$LOCAL_ROOT" status --short | sed 's/^/  /' || true
}

case "$MODE" in
    --init)   init_remote && push_code ;;
    --status) status ;;
    push|"")  push_code ;;
    *) echo "未知参数: $MODE"; echo "用法: sync_code.sh [--init|--status]"; exit 2 ;;
esac
