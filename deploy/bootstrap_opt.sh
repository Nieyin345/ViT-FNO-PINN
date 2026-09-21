#!/bin/bash
# =====================================================================
# 服务器环境部署 —— 装到 /opt（会被镜像捕获）
#
# 为什么是 /opt：
#   节点可生成镜像，但**家目录内容不会被保存**（官方警告），且依赖具体 home
#   路径会让镜像不可移植。所以 conda 与所有 python 环境装到 /opt/ 下。
#
#   镜像捕获：/opt（conda+torch）、/usr（NVIDIA 驱动，apt 装的）
#   NFS 持久：/proj（数据，见 push_data.sh）
#
# 用法（在服务器上）: bash bootstrap_opt.sh 2>&1 | tee /tmp/bs.log
# =====================================================================
set -u
CONDA_PREFIX="/opt/miniconda3"
ENV_NAME="fno"
MC_SH="/tmp/Miniconda3-latest-Linux-x86_64.sh"

step() { echo; echo "===================================================="; echo "[$(date -u '+%H:%M:%S')] $*"; echo "===================================================="; }

# ---------------------------------------------------------------- conda
step "1/3 安装 miniconda 到 $CONDA_PREFIX"
if [ -d "$CONDA_PREFIX" ]; then
    echo "已存在，跳过"
else
    curl -fsSL -o "$MC_SH" \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        || { echo "!!! 下载失败"; exit 1; }
    sudo bash "$MC_SH" -b -p "$CONDA_PREFIX" || { echo "!!! 安装失败"; exit 1; }
    sudo chown -R "$(id -u):$(id -g)" "$CONDA_PREFIX"
    rm -f "$MC_SH"
fi

source "$CONDA_PREFIX/etc/profile.d/conda.sh" || { echo "!!! conda.sh 失败"; exit 1; }
echo "conda $(conda --version)"

# 新版 miniconda 非交互下单必须显式接受 ToS，否则 conda create 直接报错
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

step "2/3 创建 conda 环境 '$ENV_NAME'"
if conda env list | grep -qE "^$ENV_NAME "; then
    echo "已存在，跳过"
else
    conda create -y -n "$ENV_NAME" python=3.11 || { echo "!!! create 失败"; exit 1; }
fi
conda activate "$ENV_NAME" || { echo "!!! activate 失败"; exit 1; }

step "3/3 安装 torch(cu121, 支持 V100) 与依赖 —— 最耗时的一步"
python -V
pip install -q --upgrade pip 2>&1 | tail -2
# V100S = Volta, compute capability 7.0。cu121 的 wheel 覆盖它。
pip install torch --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -8
pip install -q "numpy>=1.24" "scipy>=1.10" "tqdm>=4.60" "matplotlib>=3.7" "h5py>=3.8" 2>&1 | tail -4

step "自检"
python - <<'PY'
import sys, torch
print("python", sys.version.split()[0])
print("torch ", torch.__version__, "| cuda(编译)", torch.version.cuda)
print("CUDA 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {p.name}  {p.total_memory/2**30:.1f}GiB  cc{p.major}.{p.minor}")
    a = torch.randn(2000, 2000, device="cuda")
    print("  GPU 矩阵乘 OK:", torch.isfinite(a @ a).all().item())
else:
    print("  [!!] torch 看不到 CUDA")
import numpy, scipy, matplotlib, tqdm, h5py
print("numpy", numpy.__version__, "| scipy", scipy.__version__,
      "| matplotlib", matplotlib.__version__, "| h5py", h5py.__version__)
PY
echo
echo "完成。conda 前缀: $CONDA_PREFIX   环境: $ENV_NAME"
