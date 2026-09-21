#!/bin/bash
# =====================================================================
# 在服务器上启动训练（本地发起）。把产出写到持久盘 /proj，并记日志。
#
# 用法:
#   bash deploy/run_train.sh -e fno_attention --epochs 50
#   bash deploy/run_train.sh -e fno_v2 --epochs 200 --tag gen1_v2
#   bash deploy/run_train.sh --fg -e fno_attention --epochs 2   # 前台（调试）
#   bash deploy/run_train.sh --sync -e fno_attention ...         # 先同步代码再跑
#
# 说明：训练默认后台跑（nohup + 日志），因为一次 1000 epoch 会很久。
#       日志在 $REMOTE_ROOT/logs/ 下，可用 deploy/tail_train.sh 跟。
# =====================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$HERE/config.sh"

FOREGROUND=0
DO_SYNC=0
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --fg)     FOREGROUND=1; shift ;;
        --sync)   DO_SYNC=1; shift ;;
        *)        EXTRA+=("$1"); shift ;;
    esac
done

[ ${#EXTRA[@]} -eq 0 ] && { echo "用法: bash deploy/run_train.sh [-e 实验名] [--epochs N] [--tag 名字]"; exit 2; }

if [ "$DO_SYNC" = "1" ]; then
    bash "$HERE/sync_code.sh" || exit 1
fi

# 用时间戳区分每次运行，避免互相覆盖
STAMP="$(date -u '+%m%d_%H%M%S')"
TAG_ARG=""
for ((i=0; i<${#EXTRA[@]}; i++)); do
    [ "${EXTRA[$i]}" = "--tag" ] && TAG_ARG="${EXTRA[$((i+1))]}"
done
SAVE_DIR="$REMOTE_RUNS/${TAG_ARG:-run}_$STAMP"
LOG="$REMOTE_ROOT/logs/train_${TAG_ARG:-run}_$STAMP.log"

rsh "mkdir -p '$REMOTE_RUNS' '$REMOTE_ROOT/logs'"

# 把数据目录指到持久盘：train.py 读 configs/base.py 的 DATA_DIR，而它默认是
#   仓库内的 data/data。所以这里在服务器工作区建一个 data/data 软链 -> 持久数据目录。
rsh "mkdir -p '$REMOTE_REPO/data' && \
     ln -sfn '$REMOTE_DATA' '$REMOTE_REPO/data/data' && \
     ls -ld '$REMOTE_REPO/data/data'"

CMD="python -u -m tools.train ${EXTRA[*]} --save-dir '$SAVE_DIR'"

echo "===================================================================="
echo "训练: ${EXTRA[*]}"
echo "产出: $SAVE_DIR"
echo "日志: $LOG"
echo "===================================================================="

if [ "$FOREGROUND" = "1" ]; then
    # shellcheck disable=SC2086
    rsh "source '$REMOTE_ENV/miniconda3/etc/profile.d/conda.sh' && conda activate $CONDA_ENV && \
         cd '$REMOTE_REPO' && export KMP_DUPLICATE_LIB_OK=TRUE && \
         echo 'DATA_DIR 校验:' && python -c 'from configs import base as C; import os; print(\" \", C.DATA_DIR, os.path.isdir(C.DATA_DIR))' && \
         $CMD 2>&1 | tee '$LOG'"
else
    # shellcheck disable=SC2086
    rsh "source '$REMOTE_ENV/miniconda3/etc/profile.d/conda.sh' && conda activate $CONDA_ENV && \
         cd '$REMOTE_REPO' && export KMP_DUPLICATE_LIB_OK=TRUE && \
         nohup $CMD > '$LOG' 2>&1 & \
         echo \$! > '$LOG.pid'; \
         sleep 3; \
         echo '已在后台启动，PID:' \$(cat '$LOG.pid'); \
         echo '前 20 行日志:'; head -20 '$LOG' 2>/dev/null | sed 's/^/  /'"
    echo
    echo "跟进日志:  bash deploy/tail_train.sh"
    echo "拉回结果:  bash deploy/pull_results.sh"
fi
