#!/bin/bash
# =====================================================================
# 服务器环境部署（在服务器上运行，不是本地！）
#
# CloudLab 实验节点每次重启会清空本地盘 /，所以本脚本要能反复跑。
# 设计原则：**能放进 /proj（持久）的都放进去，只留必须重装的在本地盘**。
#
#   本地盘 / 重建 : NVIDIA 驱动（内核模块，必须匹配当前内核）
#   /proj 持久   : miniconda + conda 环境 + 数据 + 代码 + 结果
#
# 因此第二次以后重跑，驱动装完就结束，conda 环境直接复用。
#
# 用法（本地）:
#   bash deploy/bootstrap.sh              # 全量引导（幂等）
#   bash deploy/bootstrap.sh --check      # 只体检，不改动
#   bash deploy/bootstrap.sh --driver     # 只修驱动（重启后最常用）
# =====================================================================
set -uo pipefail

# ---- 配置（与 config.sh 保持一致；服务器上没有本地那份，这里重写一遍）----
REMOTE_USER="qinglong"
REMOTE_ROOT="/proj/rdmaopt-PG0/$REMOTE_USER/transformer"
REMOTE_ENV="$REMOTE_ROOT/env"
REMOTE_DATA="$REMOTE_ROOT/data"
REMOTE_REPO="$REMOTE_ROOT/repo"
REMOTE_RUNS="$REMOTE_ROOT/runs"
CONDA_ENV="fno"
# V100S = Volta (compute capability 7.0)。cu121 轮子支持它。
TORCH_SPEC="torch --index-url https://download.pytorch.org/whl/cu121"

MODE="${1:-all}"
MC_DIR="$REMOTE_ENV/miniconda3"

hr() { echo "----------------------------------------------------------------"; }
step() { hr; echo "[$(date -u '+%H:%M:%S')] $*"; hr; }

# =====================================================================
# 1) NVIDIA 驱动（唯一必须每次重装的部分）
# =====================================================================
check_driver() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

install_driver() {
    if check_driver; then
        echo "[驱动] 已就绪，跳过"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
        return 0
    fi

    echo "[驱动] 未检测到，开始安装"
    # apt 的包索引在重启后是空的，必须先 update。
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq || {
        echo "!!! apt update 失败"; return 1; }

    # ★ CloudLab 镜像里 dkms 是被 apt-mark hold 的（见 `apt-mark showhold`）。
    #   如果在安装列表里带上 dkms，apt 会试图升级它、撞上 hold 而整体失败：
    #       "Held packages were changed and -y was used without --allow-change-held-packages"
    #   而 dkms / 内核头文件 / gcc 在这个镜像里**本来就齐全**（实测：
    #   dkms 2.8.7、/usr/src/linux-headers-$(uname -r)、gcc、make 都在）。
    #   所以这里只装驱动本身，不碰 dkms。缺什么再单独补，避免误升级 held 包。
    local need=""
    command -v dkms  >/dev/null 2>&1 || need="$need dkms"
    command -v gcc   >/dev/null 2>&1 || need="$need build-essential"
    [ -d "/usr/src/linux-headers-$(uname -r)" ] || need="$need linux-headers-$(uname -r)"
    if [ -n "$need" ]; then
        echo "[驱动] 补齐前置: $need"
        # shellcheck disable=SC2086
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $need || {
            echo "!!! 前置安装失败"; return 1; }
    else
        echo "[驱动] 前置（dkms/headers/gcc）已齐全"
    fi

    # nvidia-driver-535-server 在 Ubuntu 22.04 + V100S 上稳妥。
    # fallback 到 ubuntu-drivers autoinstall（自动挑推荐版本）。
    if ! sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
            nvidia-driver-535-server; then
        echo "[驱动] 535-server 失败，改用 ubuntu-drivers 自动选版"
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ubuntu-drivers-common
        sudo ubuntu-drivers autoinstall || { echo "!!! 驱动安装失败"; return 1; }
    fi

    # 加载内核模块。
    #
    # ★ 关键坑：CloudLab 镜像里 nouveau（开源驱动）**开机就占住了两块 GPU**
    #   （`lspci -nnk` 显示 "Kernel driver in use: nouveau"）。专有模块这时会报
    #       modprobe: ERROR: could not insert 'nvidia': Device or resource busy
    #   dmesg 里是 "NVRM: No NVIDIA devices probed / probe routine was not called"。
    #   只写 blacklist 是没用的——那是开机时生效，**当前这次** nouveau 已经绑上了，
    #   而且 rmmod 后内核会立刻重新绑定它。
    #
    #   正确做法：先通过 sysfs 把 GPU 从 nouveau 上 **unbind**，再 rmmod，
    #   之后 modprobe nvidia 才能接管（无需重启节点）。
    unbind_from_nouveau() {
        local bound=0
        for dev in 0000:21:00.0 0000:e2:00.0; do
            local u=/sys/bus/pci/drivers/nouveau/unbind
            [ -e "$u" ] || continue
            local cur=""
            [ -e "/sys/bus/pci/devices/$dev/driver" ] && \
                cur=$(basename "$(readlink "/sys/bus/pci/devices/$dev/driver")")
            if [ "$cur" = "nouveau" ]; then
                echo "$dev" | sudo tee "$u" >/dev/null 2>&1 && \
                    echo "  $dev 已从 nouveau 解绑" && bound=1
            fi
        done
        [ "$bound" = "1" ] && sudo rmmod nouveau 2>/dev/null || true
    }

    sudo modprobe nvidia 2>/dev/null || {
        echo "[驱动] nvidia 加载失败，尝试从 nouveau 接管"
        unbind_from_nouveau
        sudo modprobe nvidia 2>/dev/null
    }
    sudo modprobe nvidia_uvm 2>/dev/null || true
    # nvidia_drm 只服务图形显示，冲突时可跳过（对训练无影响）
    sudo modprobe nvidia_drm 2>/dev/null || true

    if check_driver; then
        echo "[驱动] 安装成功"
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    else
        echo "!!! 驱动装了但 nvidia-smi 仍不可用"
        echo "    常见原因：nouveau 占着 GPU（见上面 unbind_from_nouveau 的说明），"
        echo "    或需要重启节点后再跑: bash deploy/bootstrap.sh --driver"
        return 1
    fi
}

# =====================================================================
# 2) miniconda + conda 环境（放 /proj，重启后复用）
# =====================================================================
install_conda() {
    mkdir -p "$REMOTE_ENV"

    if [ -d "$MC_DIR" ]; then
        echo "[conda] 已存在，跳过安装: $MC_DIR"
    else
        echo "[conda] 下载安装到 $MC_DIR"
        local sh="$REMOTE_ROOT/Miniconda3-latest-Linux-x86_64.sh"
        curl -fsSL -o "$sh" \
            https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
            || { echo "!!! miniconda 下载失败"; return 1; }
        bash "$sh" -b -p "$MC_DIR" || { echo "!!! miniconda 安装失败"; return 1; }
        rm -f "$sh"
    fi

    # shellcheck disable=SC1091
    source "$MC_DIR/etc/profile.d/conda.sh" || { echo "!!! conda.sh 加载失败"; return 1; }

    # 新版 miniconda 要求显式接受 repo.anaconda.com 的频道 ToS，否则
    # `conda create` 直接报 CondaToSNonInteractiveError 而失败（非交互环境下）。
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

    if conda env list | grep -qE "^$CONDA_ENV "; then
        echo "[conda] 环境 '$CONDA_ENV' 已存在，跳过创建"
    else
        echo "[conda] 创建环境 '$CONDA_ENV' (python 3.11)"
        conda create -y -n "$CONDA_ENV" python=3.11 || {
            echo "!!! conda create 失败"; return 1; }
    fi

    conda activate "$CONDA_ENV" || { echo "!!! 激活失败"; return 1; }

    echo "[conda] 安装依赖"
    # 项目 requirements: torch numpy scipy tqdm matplotlib h5py
    # torch 走 cu121 专用 index（V100 = Volta, cc 7.0）
    pip install -q --upgrade pip
    pip install -q $TORCH_SPEC || { echo "!!! torch 安装失败"; return 1; }
    pip install -q "numpy>=1.24" "scipy>=1.10" "tqdm>=4.60" \
                    "matplotlib>=3.7" "h5py>=3.8" || {
        echo "!!! 其余依赖安装失败"; return 1; }
}

# =====================================================================
# 3) 体检
# =====================================================================
do_check() {
    step "环境体检"
    local ok=1

    echo "--- 驱动 / GPU ---"
    if check_driver; then
        nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used \
                   --format=csv,noheader
    else
        echo "  [缺失] NVIDIA 驱动未就绪"; ok=0
    fi

    echo
    echo "--- conda / torch ---"
    if [ -d "$MC_DIR" ]; then
        # shellcheck disable=SC1091
        source "$MC_DIR/etc/profile.d/conda.sh"
        if conda env list | grep -qE "^$CONDA_ENV "; then
            conda activate "$CONDA_ENV"
            python - <<'PY' || ok=0
import sys
try:
    import torch
    print(f"  python {sys.version.split()[0]}")
    print(f"  torch  {torch.__version__}")
    print(f"  cuda 编译版本 {torch.version.cuda}  可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU{i}: {p.name}  {p.total_memory/2**30:.1f} GiB  cc{p.major}.{p.minor}")
    else:
        print("  [警告] torch 看不到 CUDA —— 驱动没装好，或 torch 装成了 CPU 版")
        sys.exit(1)
    import numpy, scipy, matplotlib, tqdm, h5py
    print(f"  numpy {numpy.__version__}  scipy {scipy.__version__}  "
          f"matplotlib {matplotlib.__version__}  h5py {h5py.__version__}")
except Exception as e:
    print(f"  [失败] {type(e).__name__}: {e}")
    sys.exit(1)
PY
        else
            echo "  [缺失] conda 环境 '$CONDA_ENV' 不存在"; ok=0
        fi
    else
        echo "  [缺失] miniconda 未安装"; ok=0
    fi

    echo
    echo "--- 目录 ---"
    for d in "$REMOTE_ROOT" "$REMOTE_DATA" "$REMOTE_REPO" "$REMOTE_RUNS"; do
        if [ -d "$d" ]; then
            printf "  [有] %-52s %s\n" "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)"
        else
            printf "  [无] %s\n" "$d"
        fi
    done

    echo
    echo "--- 数据 ---"
    if [ -d "$REMOTE_DATA" ]; then
        echo "  形状类别: $(ls "$REMOTE_DATA" 2>/dev/null | grep -v '\.' | tr '\n' ' ')"
        echo "  样本总数: $(find "$REMOTE_DATA" -name 'field_volume_merged.npy' 2>/dev/null | wc -l)"
    fi

    echo
    echo "--- 磁盘 ---"
    df -h "$REMOTE_ROOT" | tail -1

    hr
    if [ "$ok" = "1" ]; then echo "体检通过：可以训练"; else echo "体检未通过：见上面的 [缺失]/[失败]"; fi
    return $((1 - ok))
}

# =====================================================================
# 主流程
# =====================================================================
case "$MODE" in
    --check)
        do_check; exit $?
        ;;
    --driver)
        install_driver || exit 1
        do_check; exit $?
        ;;
    all|"")
        step "1/3 NVIDIA 驱动"
        install_driver || { echo "驱动失败，后续跳过"; exit 1; }
        step "2/3 conda 环境"
        install_conda || exit 1
        step "3/3 体检"
        do_check
        ;;
    *)
        echo "未知参数: $MODE"; echo "用法: bootstrap.sh [--check|--driver]"; exit 2
        ;;
esac
