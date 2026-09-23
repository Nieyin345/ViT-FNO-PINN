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
# ★ 为什么是**逐样本**、而不是逐形状：
#   节点的 SSH 有间歇性断连（实测同一天里多次 `port 22: Connection timed out`）。
#   逐形状传时，一个 2G 的形状断在 99% 就要**整个重传**；而且用"远端样本数
#   >= 本地样本数"做跳过判据，会把"传了 129/200"误判成"已完成"从而**永远补不齐**。
#   现在按 (样本名, 文件大小) 逐个比对，只传缺的，断在哪里都能续。
#
#   校验用**名字+大小**而不是不比对内容：本地 2G/形状，逐个 md5 要读 16G，
#   只在传输后对刚传的那几个样本校验大小。大小不符 = 上次断在写盘中途，重传即可。
#
# 用法:
#   bash deploy/push_data.sh              # 全量（8 个形状，~16G），已传的自动跳过
#   bash deploy/push_data.sh --subset     # 只传 2 个形状（冒烟）
#   bash deploy/push_data.sh --check      # 只看两边样本数
#   bash deploy/push_data.sh --force      # 忽略远端已有，全部重传
#   bash deploy/push_data.sh --batch 10   # 每 10 个样本一批（默认 20），断连代价更小
# =====================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$HERE/config.sh"

MODE="full"; FORCE=0; BATCH=20
for arg in "$@"; do
    case "$arg" in
        --force)  FORCE=1 ;;
        --subset) MODE="--subset" ;;
        --check)  MODE="--check" ;;
        full)     MODE="full" ;;
    esac
done
# --batch N：取 --batch 的下一个参数
prev=""
for arg in "$@"; do
    [ "$prev" = "--batch" ] && BATCH="$arg"
    prev="$arg"
done

hr() { echo "----------------------------------------------------------------"; }

ALL_SHAPES=(circle circlering cross cross_complex double_rect open_ring rect_ring rectangle)
SUBSET_SHAPES=(rectangle circle)

# 样本目录 = 含 field_volume_merged.npy 的目录（= 一个样本）
list_local_samples() {
    local s="$1"
    find "$LOCAL_DATA/$s" -mindepth 1 -maxdepth 1 -type d \
         -exec test -e '{}/field_volume_merged.npy' ';' -print 2>/dev/null \
        | xargs -r -n1 basename | sort
}

# 远端样本名 -> 文件字节数。一次 ssh 拿全，避免逐样本 ssh（在这条链路上很慢/易断）
# ★ 必须带重试：链路会 `Connection timed out`，一次失败就让整个形状误判为"全缺"，
#   更糟的是 `set -uo pipefail` 下调用方可能直接退出（实测踩到，脚本中途就没了）。
remote_sizes() {
    local s="$1" out n=0
    while :; do
        out=$($SSH "for d in '$REMOTE_DATA/$s'/*/; do
              [ -e \"\$d/field_volume_merged.npy\" ] || continue
              echo \"\$(basename \$d) \$(stat -c %s \"\$d/field_volume_merged.npy\")\"
          done" 2>/dev/null)
        # 成功判据：能拿到输出，或远端确实还没有该形状（两者都算"查到了"）
        if [ -n "$out" ] || $SSH "test -d '$REMOTE_DATA/$s'" 2>/dev/null; then
            echo "$out"; return 0
        fi
        n=$((n + 1))
        [ "$n" -ge 5 ] && { echo "!!! remote_sizes($s) 连不上" >&2; return 1; }
        sleep 8
    done
}

remote_count() {
    $SSH "find '$REMOTE_DATA' -name 'field_volume_merged.npy' 2>/dev/null | wc -l"
}
local_count() {
    local n=0
    for s in "${ALL_SHAPES[@]}"; do
        [ -d "$LOCAL_DATA/$s" ] && n=$((n + $(list_local_samples "$s" | wc -l)))
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
        [ -d "$LOCAL_DATA/$s" ] && printf "  %-16s %s\n" "$s" "$(list_local_samples "$s" | wc -l)"
    done
}

# 传一组样本（一次 tar + 一次 ssh，整批流式过管道）。
# ★ 关键：**一次 ssh 传整批**，而不是每个样本一次 ssh。
#   每个样本单独 tar|ssh 时，ssh 建连/认证的固定开销摊在 8MB 上，
#   实测带宽掉到 ~0.8MB/s（16G 要 5 小时）。整批一个管道后，
#   带宽由链路本身决定。
#
# 链路实测很不稳（同一天里多次 `Connection timed out` / `Connection reset`），
# 所以重试既是**必须**也是**安全**的：tar 覆写同名文件，重传只是幂等重做。
send_chunk() {
    local s="$1"; shift
    local n=0
    until tar -C "$LOCAL_DATA/$s" -cf - "$@" | \
          $SSH "tar -C '$REMOTE_DATA/$s' -xf -"; do
        n=$((n + 1))
        [ "$n" -ge 3 ] && return 1
        echo "     (传输中断，8s 后整批重传，第 $n 次)"
        sleep 8
    done
    return 0
}

push_shape() {
    local s="$1"
    echo
    echo ">>> $s"

    local -a local_names
    mapfile -t local_names < <(list_local_samples "$s")
    local total=${#local_names[@]}
    if [ "$total" -eq 0 ]; then echo "  [跳过] 本地没有样本"; return 0; fi

    # 远端 (名字 -> 大小)
    local rlist; rlist="$(remote_sizes "$s")"

    local -a todo=()
    local done_n=0
    for name in "${local_names[@]}"; do
        local lsz rsz
        lsz=$(stat -c %s "$LOCAL_DATA/$s/$name/field_volume_merged.npy" 2>/dev/null || echo 0)
        rsz=$(echo "$rlist" | awk -v n="$name" '$1==n {print $2}' | head -1)
        if [ "$FORCE" -eq 0 ] && [ -n "$rsz" ] && [ "$rsz" = "$lsz" ]; then
            done_n=$((done_n + 1))
            continue
        fi
        todo+=("$name")
    done

    echo "  已有 $done_n / $total，待传 ${#todo[@]}"
    [ ${#todo[@]} -eq 0 ] && { echo "  [完整] 全部已就绪"; return 0; }

    mkdir -p "$REMOTE_DATA/$s" 2>/dev/null
    $SSH "mkdir -p '$REMOTE_DATA/$s'" 2>/dev/null

    local ok=0 fail=0 i=0
    # 分批：一次 tar|ssh 传整批（BATCH 个样本），断连最多损失一批
    while [ $i -lt ${#todo[@]} ]; do
        local chunk=("${todo[@]:i:BATCH}")
        local n=${#chunk[@]}
        echo "  批次 [$((i+1))-$((i+n))/${#todo[@]}] ..."
        if send_chunk "$s" "${chunk[@]}"; then
            ok=$((ok + n))
        else
            fail=$((fail + n))
            echo "     !!! 本批失败（重跑本脚本会从这批继续；已成功的前缀按名+大小跳过）"
        fi
        i=$((i + n))
    done

    # 传输后校验：比对刚传的这批文件大小
    local post; post="$(remote_sizes "$s")"
    local bad=0
    for name in "${local_names[@]}"; do
        local lsz rsz
        lsz=$(stat -c %s "$LOCAL_DATA/$s/$name/field_volume_merged.npy" 2>/dev/null || echo 0)
        rsz=$(echo "$post" | awk -v n="$name" '$1==n {print $2}' | head -1)
        [ "$rsz" = "$lsz" ] || { bad=$((bad + 1)); [ $bad -le 3 ] && echo "     [大小不符] $name (本地 $lsz / 远端 ${rsz:-缺})"; }
    done
    echo "  完成: 本次成功 $ok / 失败 $fail；全形状完整 $((total - bad))/$total"
    [ "$fail" -gt 0 ] || [ "$bad" -gt 0 ] && return 1
    return 0
}

push_data() {
    local shapes=("$@")
    $SSH "mkdir -p '$REMOTE_DATA'" >/dev/null 2>&1 || { echo "!!! 建目录失败（连不上？）"; return 1; }

    hr; echo "上传 -> $REMOTE_DATA"; hr
    echo "形状: ${shapes[*]}   (每批 $BATCH 个样本$([ $FORCE -eq 1 ] && echo '，FORCE 全量重传'))"

    local bad_shapes=0
    for s in "${shapes[@]}"; do
        [ -d "$LOCAL_DATA/$s" ] || { echo ">>> $s [跳过]（本地没有）"; continue; }
        push_shape "$s" || bad_shapes=$((bad_shapes + 1))
    done

    hr
    echo "服务器样本总数: $(remote_count) / 本地 $(local_count)"
    echo "注：数据集无 field_plane_d0.10.npy 缓存，训练会自动走 v2 体数据路径（实时插值），无需预处理。"
    [ "$bad_shapes" -gt 0 ] && { echo "!!! 有 $bad_shapes 个形状未传完，重跑本脚本即可续传（已传的按名+大小跳过）"; return 1; }
    echo "全部就绪。"
    return 0
}

case "$MODE" in
    --check)  do_check ;;
    --subset) push_data "${SUBSET_SHAPES[@]}" ;;
    full|"")  push_data "${ALL_SHAPES[@]}" ;;
    *) echo "未知参数: $MODE"; exit 2 ;;
esac
