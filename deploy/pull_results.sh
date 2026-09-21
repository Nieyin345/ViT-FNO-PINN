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
    # 服务器上产出在 /proj（持久）与 /opt/repo（若有人忘了 --save-dir 的兜底）
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
    # 产出分布在两处（/proj 的 runs、/opt 的 repo/checkpoints），所以直接用
    # 绝对路径打包，不靠 cd。tar 用 -P 保留绝对路径会被去掉前导 /，故改用
    # 相对「公共父目录」不可行——这里改为逐个目录打包并解到 DEST 下同名子目录。
    for pair in "$REMOTE_RUNS:runs" "$REMOTE_REPO/checkpoints:checkpoints" "$REMOTE_REPO/runs:repo_runs"; do
        RDIR="${pair%%:*}"; SUB="${pair##*:}"
        if $SSH "[ -d '$RDIR' ]" 2>/dev/null; then
            echo "  取 $RDIR"
            mkdir -p "$DEST/$SUB"
            $SSH "tar -C '$RDIR' -cf - . 2>/dev/null" | tar -C "$DEST/$SUB" -xf - 2>/dev/null
        fi
    done

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
