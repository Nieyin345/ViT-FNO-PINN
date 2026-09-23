#!/bin/bash
# =====================================================================
# 共享配置：所有 deploy/*.sh 都 source 本文件。改节点只改这里。
# =====================================================================

# ---- 目标节点 ----
# 注：CloudLab 实验重启后会换 hostname，届时改 NODE 一行即可。
# 2026-09-23: Clemson 的 GPU 实验（clgpu016/017）连不上——密钥没登记到那批节点，
#   换到 Wisconsin 新实验（qinglong-317487）。硬件也换了：
#   V100S×2 (Volta sm_70) -> **Tesla P100 PCIe 12GB ×1 (Pascal sm_60)**。
#   sm_60 在 torch cu121 的 wheel 覆盖范围内，不用换 torch。
NODE="c240g5-110205.wisc.cloudlab.us"
REMOTE_USER="qinglong"

# ---- SSH ----
# `IdentitiesOnly yes` 是必须的：Git Bash 自带的 ssh 摸不到 Windows ssh-agent，
# 不写这行它会先试 agent 里的 key（拿不到）就放弃，表现为
# "Permission denied (publickey)"，而 Windows OpenSSH 却能连上。
KEY="$HOME/.ssh/qkd_rl/qkd_rl_deploy"
SSH_OPTS="-o IdentitiesOnly=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new"
SSH="ssh $SSH_OPTS -i $KEY $REMOTE_USER@$NODE"
SCP="scp $SSH_OPTS -i $KEY"

# ---- 服务器上的目录布局 ----
# ★ 关键：节点**可以生成镜像**，但官方明确警告
#     "The contents of your home directory is NOT saved and will be deleted
#      during the imaging process"
#   所以软件装到**标准位置** /opt（会被镜像捕获），数据放 /proj（NFS 持久）。
#
#   镜像捕获  : /opt  (miniconda + fno 环境 + 代码)
#               /usr  (NVIDIA 驱动，apt 装的)
#   NFS 持久  : /proj (数据集，不必打进镜像——16G 太大且非软件)
#   家目录    : 不放任何东西（会被清）
OPT_ROOT="/opt/transformer"
OPT_CONDA="/opt/miniconda3"          # miniconda（镜像捕获）
REMOTE_BARE="$OPT_ROOT/repo.git"     # 裸仓库：代码同步的中间站
REMOTE_REPO="$OPT_ROOT/repo"         # 工作区：训练在这里跑
# 数据放 /proj：16G，不必进镜像；且 /proj 是持久的
REMOTE_DATA="/proj/rdmaopt-PG0/$REMOTE_USER/transformer/data"
REMOTE_RUNS="/proj/rdmaopt-PG0/$REMOTE_USER/transformer/runs"
REMOTE_LOGS="/proj/rdmaopt-PG0/$REMOTE_USER/transformer/logs"

# ---- conda 环境 ----
CONDA_ENV="fno"

# ---- 本地 ----
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DATA="D:/destop/hfss-new/meep_out"

# ---- 便捷函数 ----
rsh() { $SSH "$@"; }

# 在服务器上以 fno 环境运行命令
rr() {
    $SSH "source $OPT_CONDA/etc/profile.d/conda.sh && conda activate $CONDA_ENV && cd $REMOTE_REPO && $*"
}

hr() { echo "----------------------------------------------------------------"; }
