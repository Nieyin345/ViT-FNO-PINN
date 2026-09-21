#!/bin/bash
# =====================================================================
# 数据上传（本地 -> 服务器）。16GB，只需**传一次**。
#
# 放 /proj（项目 NFS，持久）：不随镜像走（16G 太大，且是数据不是软件），
# 但 /proj 本身跨实验重启持久，所以正常只需传一次。
#
# 为什么用 tar 而不是 rsync：本地 Git Bash **没有 rsync**（实测 which rsync 为空），
# 服务器有。所以用 `tar | ssh tar -x` 流式传输，不落中间文件。
#
# 用法:
#   bash deploy/push_data.sh            # 全量（8 个形状，~16G）
#   bash deploy/push_data.sh --subset   # 只传 2 个形状（冒烟）
#   bash deploy/push_data.sh --check    # 只看两边样本数
# =====================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$HERE/config.sh"

MODE="${1:-full}"
hr() { echo "----------------------------------------------------------------"; }

ALL_SHAPES=(circle circlering cross cross_complex double_rect open_ring rect_ring rectangle)
SUBSET_SHAPES=(rectangle circle)

remote_count() {
    $SSH "find '$REMOTE_DATA' -name 'field_volume_merged.npy' 2>/dev/null | wc -l"
}
local_count() {
    local n=0
    for s in "${ALL_SHAPES[@]}"; do
        [ -d "$LOCAL_DATA/$s" ] && n=$((n + $(find "$LOCAL_DATA/$s" -name 'field_volume_merged.npy' | wc -l)))
    done
    echo "$n"
}

do_check() {
    hr; echo "数据体检"; hr
    echo "本地  ($LOCAL_DATA): $(local_count) 个样本"
    $SSH "mkdir -p '$REMOTE_DATA'" 2>/dev/null
    echo "服务器 ($REMOTE_DATA): $(remote_count)"
    hr; echo "本地各形状样本数:"; hr
    for s in "${ALL_SHAPES[@]}"; do
        [ -d "$LOCAL_DATA/$s" ] && printf "  %-16s %s\n" "$s" "$(find "$LOCAL_DATA/$s" -name 'field_volume_merged.npy' | wc -l)"
    done
}

push_data() {
    local shapes=("$@")
    $SSH "mkdir -p '$REMOTE_DATA'" || { echo "!!! 建目录失败"; return 1; }

    hr; echo "上传 -> $REMOTE_DATA"; hr
    echo "形状: ${shapes[*]}"
    echo "（tar over ssh 流式传输，大文件耐心等）"

    local args=()
    for s in "${shapes[@]}"; do
        [ -d "$LOCAL_DATA/$s" ] && args+=("$s")
    done
    [ ${#args[@]} -eq 0 ] && { echo "!!! 本地没找到指定形状"; return 1; }

    # -C 切到父目录：解包后直接落在 $REMOTE_DATA/<shape>/ 下
    tar -C "$LOCAL_DATA" -cf - "${args[@]}" \
        | $SSH "tar -C '$REMOTE_DATA' -xf -" || { echo "!!! 传输失败"; return 1; }

    hr
    echo "完成。服务器样本数: $(remote_count)"
    echo "注：数据集无 field_plane_d0.10.npy 缓存，训练会自动走 v2 体数据路径（实时插值），无需预处理。"
}

case "$MODE" in
    --check)  do_check ;;
    --subset) push_data "${SUBSET_SHAPES[@]}" ;;
    full|"")  push_data "${ALL_SHAPES[@]}" ;;
    *) echo "未知参数: $MODE"; exit 2 ;;
esac
