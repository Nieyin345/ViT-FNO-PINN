# deploy/ — 服务器训练工作流

把本仓库的训练放到 CloudLab GPU 节点上跑，本地负责改代码、看结果。

## 目标节点

| 项 | 值 |
|---|---|
| 节点 | `clgpu016.clemson.cloudlab.us`（用户 `qinglong`） |
| 硬件 | 2× Tesla V100S-PCIE-32GB、128 vCPU、503 GB RAM |
| 系统 | Ubuntu 22.04.2、内核 5.15.0-187 |
| 密钥 | `~/.ssh/qkd_rl/qkd_rl_deploy` |
| 已装 | conda 到 `/opt/miniconda3`，`fno` 环境含 torch 2.5.1+cu121 |

节点名只在 `config.sh` 的 `NODE=` 一处改。CloudLab 实验**每次重启都会换 hostname**。

## 命令

```bash
bash deploy/bootstrap_opt.sh     # 环境部署（在服务器上跑；幂等）
bash deploy/sync_code.sh --init  # 首次：建裸仓库 + 上传代码
bash deploy/sync_code.sh         # 之后：增量同步
bash deploy/push_data.sh         # 上传 16G 数据（只需一次）
bash deploy/run_train.sh -e fno_attention --epochs 50 --tag trial1 --sync
bash deploy/tail_train.sh        # 跟日志 / --gpu / --ps
bash deploy/pull_results.sh      # 结果拉回本地
```

## 核心设计：镜像把 `/opt` 带走，数据留在 `/proj`

节点**可以生成镜像**，但官方明确警告：

> The contents of your home directory is NOT saved and will be deleted during the
> imaging process. You should install your software and data files in standard
> locations like `/usr/local` or `/opt`.

所以按"是否进镜像"分两块放：

```
进镜像（重启后还在，因为是镜像的一部分）
  /opt/miniconda3        miniconda + fno 环境（torch 2.5.1+cu121）
  /opt/transformer/repo  代码工作区
  /usr                    NVIDIA 驱动（apt 装的位置）

不进镜像
  /proj/rdmaopt-PG0/qinglong/transformer/data   16G 数据集（NFS 持久，传一次）
  /proj/.../runs、logs                           训练产出（NFS 持久）

绝不落地
  ~/                      家目录不入镜像，别往里放东西
```

## 目录布局（服务器）

```
/opt/transformer/
├── repo.git/     # 裸仓库：本地 git push 到这里
└── repo/         # 工作区：从裸仓库检出，训练在这跑
/proj/rdmaopt-PG0/qinglong/transformer/
├── data/         # 数据集（8 形状 × 200，约 16G）
├── runs/         # 训练产出
└── logs/         # nohup 训练日志
```

## 为什么代码用 git、数据用 tar

本地 Git Bash **没有 rsync**（实测 `which rsync` 为空），服务器有。
所以代码走 git（增量、有历史），16G 数据走 `tar | ssh tar -x`（流式，不落中间文件）。

代码同步是 **本地 → 裸仓库 → 工作区**，而不是直推工作区：

```
本地 repo  --git push-->  /opt/transformer/repo.git  --checkout-->  /opt/transformer/repo
```

工作区永远是干净检出（本地怎么折腾都不影响它），第二次起 push 只传增量。

## 踩过的坑（都已在脚本里处理）

1. **`git push` 不认 config.sh 里的 ssh 选项**
   `git push` 自己拉起 ssh，于是踩上"Git Bash 摸不到 Windows ssh-agent"那个坑，
   报 `Permission denied (publickey)`。
   必须用 `GIT_SSH_COMMAND="ssh -o IdentitiesOnly=yes -i <key>"` 显式指定
   （`sync_code.sh` 里本地 push 与服务器 clone 两处都要）。

2. **nouveau 占着 GPU**（最隐蔽）
   CloudLab 镜像开机就让开源 `nouveau` 驱动绑住两块 V100S，专有模块加载时报
   `Device or resource busy`，dmesg 是 `NVRM: No NVIDIA devices probed`。
   只写 `blacklist nouveau` **没用**——那是下次开机生效，当前这次 nouveau 已经绑上，
   且 `rmmod` 后内核会立刻重新绑定。
   正确做法：先 sysfs `unbind` 把 GPU 从 nouveau 摘下，再 `rmmod`，再 `modprobe nvidia`，
   **无需重启节点**。

3. **`dkms` 被 apt-mark hold**
   把它写进 apt 包列表会让 apt 试图升级它、撞上 hold 而**整体失败**
   （`Held packages were changed...`）。而镜像里 dkms/内核头/gcc 本来就齐全
   （dkms 2.8.7、`/usr/src/linux-headers-$(uname -r)` 都在），所以只装驱动本身。

4. **apt 索引重启后是空的** —— 必须先 `apt-get update`。

5. **miniconda 新版要显式接受 ToS**
   非交互环境下 `conda create` 会直接报 `CondaToSNonInteractiveError`，
   须先 `conda tos accept --override-channels --channel <url>`。

6. **numpy 必须钉在 1.x（服务器与本地口径一致）**
   服务器默认装出 numpy 2.4.6，torch.save 出的 checkpoint 里 pickle 记的是
   `numpy._core`，而**本地** pytorch env 是 numpy 1.24，反序列化直接报
   `ModuleNotFoundError: No module named 'numpy._core'`（实测确认）。
   `bootstrap_opt.sh` 已钉 `numpy>=1.24,<2`，装完还会复查一次
   （torch 2.5 的依赖可能把它拉回 2.x）。实测服务器 1.26.4 → 本地 1.24.3 加载正常。

7. **`IdentitiesOnly yes` 是必须的**
   Git Bash 自带的 ssh 摸不到 Windows ssh-agent，不写这行它会先试 agent 里的 key
   （拿不到）就放弃，而 Windows OpenSSH 却能连上。

8. **换行符**：本地 `core.autocrlf=true` 会把 `.sh` 转 CRLF，Linux 上报
   `bad interpreter: /bin/bash^M`。仓库根的 `.gitattributes` 已强制 `*.sh` 用 LF。

## 数据

`D:\destop\hfss-new\meep_out`，8 形状 × 200 样本 ≈ 16G：

```
circle/ circlering/ cross/ cross_complex/ double_rect/ open_ring/ rect_ring/ rectangle/
   <shape>_N/
     field_volume_merged.npy   (2, 9, 3, 64, 64, 12)  双极化体数据
     patch_plane.npy           (64, 64, 2)            SDF + Mask
     *_S_Parameters.csv        HFSS S 参数
     run.log
```

格式与 `data/dataset.py` 一致，会被识别为 **v2 体数据**路径。
数据集**没有** `field_plane_d0.10.npy` 缓存，训练会实时插值到 0.1λ —— 这是正确的，
不需要跑 `preprocess_v2.py`。

## 下一轮（训练）待办

- [ ] 上传数据（`push_data.sh`，16G，一次）
- [ ] 首次端到端训练跑通并检查 GPU 占用
- [ ] 确认 `run_train.sh` 把 `DATA_DIR` 正确指向 `/proj/.../data`
- [ ] 决定是否把当前环境**生成镜像**（省去以后每次 bootstrap）
