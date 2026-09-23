#!/bin/bash
# =====================================================================
# GPU 驱动安装（在服务器上跑，需要 sudo）。幂等。
#
# 为什么单独一个脚本：bootstrap_opt.sh 只管 conda/torch，假设驱动已就绪。
# 但 CloudLab 镜像开机会让**开源 nouveau** 绑住卡，专有模块加载时报
# `Device or resource busy` / `NVRM: No NVIDIA devices probed`。
# 只写 blacklist 没用——那是下次开机才生效，当前这次 nouveau 已经绑上了，
# 且 rmmod 后内核立刻重新绑定。正确顺序：sysfs unbind -> rmmod -> apt 装驱动
# -> modprobe nvidia。全过程**不需要重启节点**。
#
# 用法（服务器上）: sudo bash setup_gpu.sh 2>&1 | tee /tmp/gpu.log
# =====================================================================
set -u
DRIVER_VER="${1:-550}"

step() { echo; echo "===================================================="; echo "[$(date -u '+%H:%M:%S')] $*"; echo "===================================================="; }

if [ "$(id -u)" -ne 0 ]; then echo "!!! 需要 root（用 sudo 跑）"; exit 1; fi

# ----------------------------------------------------------- 1. 摘掉 nouveau
step "1/4 把 GPU 从 nouveau 摘下"
# 找出所有当前由 nouveau 驱动的 PCI 设备（P100 的地址在不同节点可能不同，动态探测）
NOUVEAU_DEVS=()
for d in /sys/bus/pci/drivers/nouveau/0000:*; do
    [ -e "$d" ] || continue
    NOUVEAU_DEVS+=("$(basename "$d")")
done

if [ ${#NOUVEAU_DEVS[@]} -eq 0 ]; then
    echo "nouveau 没有绑定任何设备（可能已经装好驱动了）"
else
    for dev in "${NOUVEAU_DEVS[@]}"; do
        echo "  unbind $dev"
        echo "$dev" > /sys/bus/pci/drivers/nouveau/unbind 2>/dev/null || echo "    (unbind 失败，继续)"
    done
fi

# 先摘依赖：nouveau 引用 ttm/drm_ttm_helper，不先卸它们 rmmod 会失败
if lsmod | grep -q '^nouveau'; then
    echo "  卸载 nouveau 模块"
    rmmod nouveau 2>/dev/null || {
        # 依赖链反过来卸
        for m in drm_ttm_helper drm_kms_helper drm; do
            rmmod "$m" 2>/dev/null || true
        done
        rmmod nouveau 2>/dev/null || echo "    (rmmod 失败，apt 装驱动时再试)"
    }
fi
echo "  lsmod 里还有 nouveau 吗: $(lsmod | grep -c '^nouveau') （0 = 干净）"

# 持久黑名单：进镜像后新节点开机也不会再绑 nouveau
echo " 写入 /etc/modprobe.d/blacklist-nouveau.conf"
cat > /etc/modprobe.d/blacklist-nouveau.conf <<'EOF'
blacklist nouveau
options nouveau modeset=0
EOF

# ----------------------------------------------------------- 2. 装驱动
step "2/4 apt 装 nvidia-driver-$DRIVER_VER（最耗时，几分钟）"
export DEBIAN_FRONTEND=noninteractive
# 索引在重启后可能是空的，先 update
apt-get update -qq || { echo "!!! apt update 失败"; exit 1; }
# ★ 不要装 dkms：它被 apt-mark hold，写进包列表会让整条 apt 命令失败
#   （Held packages were changed...）。镜像里 dkms/内核头/gcc 本来就齐全。
apt-get install -y -qq "nvidia-driver-$DRIVER_VER" || { echo "!!! 驱动安装失败"; exit 1; }

# ----------------------------------------------------------- 3. 加载
step "3/4 加载 nvidia 模块"
modprobe nvidia 2>&1 || echo "  (modprobe 报错，可能 postinst 已加载)"
# 再确认一次 nouveau 没回来
lsmod | grep -q '^nouveau' && {
    echo "  [!] nouveau 又回来了，重摘一次"
    for d in /sys/bus/pci/drivers/nouveau/0000:*; do
        [ -e "$d" ] && echo "$(basename "$d")" > /sys/bus/pci/drivers/nouveau/unbind 2>/dev/null
    done
    rmmod nouveau 2>/dev/null || true
    modprobe nvidia 2>/dev/null || true
}

# ----------------------------------------------------------- 4. 自检
step "4/4 自检"
which nvidia-smi || echo "!!! 没有 nvidia-smi"
nvidia-smi 2>&1 | head -15
echo
echo "驱动: $(cat /proc/driver/nvidia/version 2>/dev/null || echo '读不到 /proc/driver/nvidia/version')"
