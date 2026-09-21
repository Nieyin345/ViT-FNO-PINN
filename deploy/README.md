# deploy/ — 服务器训练工作流

把本仓库的训练放到 CloudLab GPU 节点上跑，本地负责改代码、看结果。

## 目标节点

| 项 | 值 |
|---|---|
| 节点 | `clgpu016.clemson.cloudlab.us`（用户 `qinglong`） |
| 硬件 | 2× Tesla V100S-PCIE-32GB、128 vCPU、503 GB RAM |
| 系统 | Ubuntu 22.04.2、内核 5.15.0-187 |
| 密钥 | `~/.ssh/qkd_rl/qkd_rl_deploy` |

节点名在 `config.sh` 的 `NODE=` 一处改。CloudLab 实验**每次重启都会换 hostname**，
届时改这一行即可。

## 立即可用的命令

```bash
bash deploy/bootstrap.sh          # 环境部署（幂等，可反复跑）
bash deploy/sync_code.sh --init   # 首次：建裸仓库 + 上传代码
bash deploy/push_data.sh          # 上传 16G 数据（只需一次）
bash deploy/run_train.sh -e fno_attention --epochs 50 --tag trial1 --sync
bash deploy/tail_train.sh         # 看训练日志
bash deploy/pull_results.sh       # 把结果拉回本地
```

## 这套脚本的核心设计：**只有驱动在会丢的那块盘上**

CloudLab 的实验节点，重启后**本地盘 `/`（63G）被清空**，但项目 NFS
**`/proj/rdmaopt-PG0` 是持久的**（实测重启后目录内容仍在）。所以：

```
本地盘 /  （会丢）  ← 只有 NVIDIA 驱动（内核模块，必须匹配当前内核）
/proj     （持久）  ← miniconda + conda 环境 + 16G 数据 + 代码 + 训练产出
```

后果：**重启后只需要重跑 `bootstrap.sh --driver`（几分钟），conda 环境和数据都还在。**
这也是为什么 `bootstrap.sh` 把 conda 装在 `/proj` 而不是默认的 `~/miniconda3`。

## 目录布局（服务器上）

```
/proj/rdmaopt-PG0/qinglong/transformer/
├── env/miniconda3/     # conda 与 fno 环境（持久）
├── data/               # meep_out 数据集，16G（持久，传一次）
├── repo.git/           # 裸仓库：本地 git push 到这里（代码同步的中间站）
├── repo/               # 工作区：从裸仓库 checkout，训练在里跑
├── runs/               # 训练产出（checkpoints/summary）
└── logs/               # nohup 训练日志
```

## 为什么代码用 git、数据用 tar

本地 Git Bash **没有 rsync**（实测 `which rsync` 为空），服务器有。
所以：代码走 git（增量、有历史、本来就是版本控制的东西），
16G 数据走 `tar | ssh tar -x`（流式，不落中间文件）。

代码同步的结构是 **本地 → 裸仓库 → 工作区**，而不是直推工作区：

```
本地 repo  --git push-->  repo.git  --checkout-->  repo/
```

这样工作区永远是干净检出，且第二次之后 push 只传增量。

## 踩过的坑（都已在脚本里处理）

1. **nouveau 占着 GPU**（最隐蔽）
   CloudLab 镜像开机就让开源的 `nouveau` 驱动绑住了两块 V100S，专有模块加载时报
   `Device or resource busy`，dmesg 里是 `NVRM: No NVIDIA devices probed`。
   只写 `blacklist nouveau` **没用**——那是下次开机生效，当前这次 nouveau 已经绑上了，
   而且 `rmmod` 后内核会立刻重新绑定。
   正确做法：先用 sysfs `unbind` 把 GPU 从 nouveau 摘下来，再 `rmmod`，
   再 `modprobe nvidia`（`bootstrap.sh` 的 `unbind_from_nouveau` 就是干这个的），**无需重启节点**。

2. **`dkms` 被 apt-mark hold**
   CloudLab 把 `dkms` 钉死了版本。安装时若把它写进 apt 包列表，apt 会试图升级它、
   撞上 hold 而**整体失败**（`Held packages were changed...`）。
   而 dkms/内核头文件/gcc 在这个镜像里本来就齐全（实测 dkms 2.8.7、
   `/usr/src/linux-headers-$(uname -r)` 都在）。所以只装驱动本身，不碰 dkms。

3. **apt 索引在重启后是空的**
   必须先 `apt-get update`，否则 `apt-cache policy <pkg>` 返回空、装不了东西。

4. **`IdentitiesOnly yes` 是必须的**
   Git Bash 自带的 ssh 摸不到 Windows ssh-agent，不写这行它会先试 agent 里的 key
   （拿不到）就放弃，表现为 `Permission denied (publickey)`，
   而 Windows OpenSSH 却能连上。所有脚本里的 ssh 都带了这个选项。

5. **换行符**
   本地 `core.autocrlf=true` 会把 `.sh` 转成 CRLF，Linux 上报
   `bad interpreter: /bin/bash^M`。仓库根的 `.gitattributes` 已强制 `*.sh` 用 LF。

## 数据

`D:\destop\hfss-new\meep_out`，8 个形状 × 200 样本 ≈ 16G：

```
circle/ circlering/ cross/ cross_complex/ double_rect/ open_ring/ rect_ring/ rectangle/
   <shape>_N/
     field_volume_merged.npy   (2, 9, 3, 64, 64, 12)  双极化体数据
     patch_plane.npy           (64, 64, 2)            SDF + Mask
     *_S_Parameters.csv        HFSS S 参数
     run.log
```

格式与 `data/dataset.py` 期望的一致，会被识别为 **v2 体数据**路径。
数据集里**没有** `field_plane_d0.10.npy` 缓存，训练会实时插值到 0.1λ —— 这是正确的，
不需要额外跑 `preprocess_v2.py`。
