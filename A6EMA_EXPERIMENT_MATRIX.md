# A6ema 公平对比与消融实验协议

更新：2026-10-03。远程 `tools.smoke` 与全矩阵 `--dry-run` 均通过。用户确认完整 9 配置 × 3 种子 × 24 epochs 后，已按每个训练种子一个 compare worker 并行；seed 1/2/3 的 `A6ema` 首项都已进入 CUDA。最新快照合计显存约 3.7/12 GiB、GPU 利用率 100%、温度 49°C。显存未满不代表算力闲置；当前 batch size 16 固定，不在矩阵中途改动。若后续要测更大 batch，应作为独立、统一预算的实验。最近日志快照：seed 1 的 A6ema 为 epoch 11、1125/1280 批次；seed 2/3 分别为 epoch 5、1139/1280 与 1134/1280。按当前吞吐估计，矩阵尚需约 17–20 小时，实际以各配置 wall-time 为准。暂无完整配置或指标汇总。worker：`run_a6ema_seed_remote.sh`；日志分别为 `logs/a6ema_full_matrix_20261003_full.log`、`logs/a6ema_matrix_seed2_20261003.log`、`logs/a6ema_matrix_seed3_20261003.log`。未在本地训练或处理数据，数据未改。

## 1. 评估口径（以源码为准）

- 主设定固定为 8 类形状：`circle, circlering, cross, cross_complex, double_rect, open_ring, rect_ring, rectangle`。
- `TRAIN_FREQS=None`、`HOLDOUT_FREQS=[10.0]`：10 GHz 样本既不进训练也不进常规验证；其他 8 个频点用于训练/验证。10 GHz 是频带内插值点，不是外推点。
- `build_split()` 将保留频点分为 `unseenF_seenS@10GHz`（训练形状，只测频率泛化）和 `unseenF_unseenS@10GHz`（形状与频率都未见）。输出标准化参数由训练子集计算，不读取 holdout 标签。
- 选模只看常规验证集 `val_geom`；代码仅在某轮成为最佳轮时评估 holdout，holdout 不参与选模。**因此 `val_geom` 是已见频率上的形状泛化，不能替代未见频率结果。**
- 旧 `*_freqhold` 运行把频率从训练样本中删除，口径与本矩阵不同；不与本矩阵混表。`A6ema_h1011` 同时留出 10、11 GHz，训练频点更少，也单列比较。

## 2. 实验矩阵

### 2.1 核心 2×2 因子消融（EMA 固定为 0.995）

| 实验 | PhaseFrame | 频率基 | 要回答的问题 |
|---|---:|---|---|
| `A6ema` | 开 | K=6 Chebyshev | 完整模型 |
| `A6ema_nophase` | 关 | K=6 Chebyshev | 相位参考系的独立贡献 |
| `A6ema_constbasis` | 开 | K=6 常数平均 | Chebyshev 频率基的独立贡献 |
| `A6ema_nophase_constbasis` | 关 | K=6 常数平均 | 两组件都关；用于估计交互 |

`constant` 模式仍保留 K=6 的全部输出系数头和参数量，所有系数头以等权常数参与合成；主体仍 `free_body=True`，所以该对照移除了**学习到的频率基依赖**，而不是把头缩小。PhaseFrame 开启时，已知物理相位旋转仍保留。源码新增的 `poly_basis_mode` 只接受 `chebyshev` 或 `constant`。

另加一组 EMA 单变量对照：`A6_noema` 与 `A6ema` 完全相同，仅 `EMA_DECAY` 从 0.995 改为 0。

### 2.2 架构对照

| 实验 | 模型 | PhaseFrame | EMA | 说明 |
|---|---|---:|---:|---|
| `A6ema_fno_baseline` | `fno_baseline` | 开 | 0.995 | 经典 FNO 基线 |
| `A6ema_fno_film` | `fno_film` | 开 | 0.995 | 多频 FiLM 对照；不是 Chebyshev 的严格单因素消融 |
| `A6ema_fno_attention` | `fno_attention` | 开 | 0.995 | 注意力 FNO 对照 |
| `A6ema_unet` | `unet` | 开 | 0.995 | 非 FNO 架构对照 |

这些实验共享类别、频率留出、损失、数据划分和训练预算。不同架构的参数量不保证一致；比较表会记录 `n_params`，不把架构对照误称为参数匹配。Chebyshev 的归因以 2×2 消融为准。

## 3. 固定训练预算与种子

固定每个模型 24 epochs、batch size 16、同一默认优化器/损失/输出归一化；`LR_WARMUP_EPOCHS=2`、`FREQ_INTERP_PROB=0`，其余默认训练项不变。24 epochs 小于早停 patience 60，因此该矩阵按相同 epoch 数完成。所有实验固定划分种子 `SHUFFLE_SEED=42`；`TRAIN_SEED` 分别为 1、2、3，用于配对的初始化/训练随机性重复。**同一个 TRAIN_SEED 下比较所有模型**，不要用 `--seed` 代替它（`--seed` 会改变划分）。

在远程仓库根目录执行；每个种子单独存结果目录：

```bash
set -e
cd /users/qinglong/transformer
# One compare worker per training seed; separate roots avoid summary/output collisions.
for s in 1 2 3; do
  nohup /bin/bash run_a6ema_seed_remote.sh "$s" \
    > "logs/a6ema_matrix_seed${s}_20261003.log" 2>&1 < /dev/null &
done
```

`tools.compare` 会为每个 run 名带上 split/train seed，保存 `scenario.split` 指纹；同一轮比较若 split 指纹或场景不一致，应先查明并修正，不能直接下结论。比较器现在也会在终端单列 holdout 的 geom/rel，并把各 holdout 组的 geom/rel/n 展开写入 `summary.csv`。

## 4. 报告与归因

1. 主要频率泛化指标：`unseenF_seenS@10GHz.geom`；同时报告 `unseenF_unseenS@10GHz.geom`。`val_geom`、`val_rel` 作为已见频率的形状泛化/相对误差补充，三者不能混称。
2. 每个 seed 内对照同一划分，报告 3 个 seed 的均值、标准差和逐 seed 配对差值；2×2 组再报告 phase×Chebyshev 的差分之差。EMA 以 `A6_noema` 对 `A6ema` 做配对差值。
3. 汇报最佳 epoch、参数量、scenario/split 指纹和全部 holdout 组。不要只挑最好 seed，也不要因单个种子差异就宣称显著；实测若无稳定提升，就如实写“未观察到稳定增益”。
4. 历史记录仅作背景，不代替本矩阵：A6ema 曾有 12 epochs、3 seeds 的 `val_geom=0.2045±0.0045`，以及 seed 1、24 epochs 的 `val_geom=0.1900`；预算/种子指纹需复核后才可并表。
