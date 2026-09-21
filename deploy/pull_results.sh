#!/bin/bash
# =====================================================================
# 结果回传（服务器 -> 本地）。训练产出在 /proj（持久），但要拿回本地看，
# 且本地做可视化/写论文时不必再登服务器。
#
# 回传内容：checkpoints/（best.pth 等）、runs/（summary.json/csv）、
#          训练日志与曲线。
#
# 用法:
#   bash deploy/pull_results.sh                 # 回传到 ./remote_results/
#   bash deploy/pull_results.sh --to D:/x       # 指定本地落地目录
#   bash deploy/pull_results.sh --list          # 只看服务器上有什么
# =====================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$HERE/config.sh"

DEST="$LOCAL_ROOT/remote_results"
MODE="pull"
while [ $# -gt 0 ]; do
    case "$1" in
        --to)   DEST="$2"; shift 2 ;;
        --list) MODE="list"; shift ;;
        *) echo "未知参数: $1"; exit 2 ;;
    esac
done
hr() { echo "----------------------------------------------------------------"; }

# 服务器上的产出分布：
#   $REMOTE_RUNS    —— 训练时把 SAVE_DIR 指到这里（见 run_train.sh）
#   $REMOTE_REPO/checkpoints, $REMOTE_REPO/runs —— 若有人忘了改 SAVE_DIR 的兜底
remote_list() {
    rsh "for d in '$REMOTE_RUNS' '$REMOTE_REPO/checkpoints' '$REMOTE_REPO/runs'; do \
             if [ -d \"\$d\" ]; then \
                 echo \"[\$d]\"; \
                 find \"\$d\" -type f \\( -name '*.pth' -o -name '*.json' -o -name '*.csv' -o -name '*.npz' -o -name '*.log' \\) \
                     -printf '  %10s  %p\n' 2>/dev/null | head -40; \
             fi; \
         done"
}

do_pull() {
    mkdir -p "$DEST"
    hr; echo "回传结果 -> $DEST"; hr
    if [ -z "$(remote_list)" ]; then
        echo "服务器上还没有产出（还没训练，或结果在别处）。"
        return 0
    fi
    remote_list

    hr; echo "打包传输"; hr
    # 服务器侧打 tar（服务器有 rsync，直接用 rsync 更稳，但为避免多一条依赖，
    # 统一用 tar over ssh，与 push_data.sh 对称）。
    rsh "cd '$REMOTE_ROOT' 2>/dev/null && \
         tar -cf - \$(for d in runs repo/checkpoints repo/runs; do [ -d \"\$d\" ] && echo \"\$d\"; done) 2>/dev/null" \
        | tar -C "$DEST" -xf - 2>/dev/null || {
        echo "!!! 传输失败"; return 1; }

    hr
    echo "已回传到: $DEST"
    find "$DEST" -type f \( -name '*.pth' -o -name '*.json' -o -name '*.csv' \) 2>/dev/null \
        | head -30 | sed 's/^/  /'
    echo
    echo "提示：本地可直接跑  python -m tools.report <dir>  汇总成表。"
}

case "$MODE" in
    list) hr; echo "服务器产出"; hr; remote_list ;;
    pull) do_pull ;;
esac
