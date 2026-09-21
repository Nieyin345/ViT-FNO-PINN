#!/bin/bash
# =====================================================================
# 跟训练日志 / 看状态。
#
# 用法:
#   bash deploy/tail_train.sh          # 最近一次训练的最后 40 行
#   bash deploy/tail_train.sh -f       # 持续跟随
#   bash deploy/tail_train.sh --gpu    # 看 GPU 占用
#   bash deploy/tail_train.sh --ps     # 看训练进程
# =====================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$HERE/config.sh"

MODE="${1:-tail}"
hr() { echo "----------------------------------------------------------------"; }

latest_log() {
    rsh "ls -t '$REMOTE_LOGS/'*.log 2>/dev/null | head -1"
}

case "$MODE" in
    --gpu)
        hr; echo "GPU"; hr
        rsh "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"
        ;;
    --ps)
        hr; echo "训练进程"; hr
        rsh "ps aux | grep '[t]ools.train' || echo '  (没有正在跑的训练)'"
        ;;
    -f|--follow)
        LOG="$(latest_log)"
        [ -z "$LOG" ] && { echo "还没有日志"; exit 0; }
        echo "跟随: $LOG  (Ctrl-C 退出)"
        rsh "tail -f '$LOG'"
        ;;
    tail|"")
        LOG="$(latest_log)"
        [ -z "$LOG" ] && { echo "还没有日志"; exit 0; }
        hr; echo "最新日志: $LOG"; hr
        rsh "tail -40 '$LOG'"
        ;;
    *) echo "未知参数: $MODE"; exit 2 ;;
esac
