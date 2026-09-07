# 电磁超表面场预测模型文档

## 项目主线（不变）

1. **patch2field**：用 FNO（Fourier Neural Operator）从贴片几何（SDF + Mask）预测**近场截面**的场分布（12 通道：Ex/Ey/Ez + Hx/Hy/Hz 的实部虚部）。
2. **field2field**：用物理传播（PWS / 角谱法，E+H 双向）把单个截面的散射场传播到任意 z 位置，重建整个空气盒子的三维场分布。
   端到端衔接脚本：`field2field/propagate_fno_prediction.py`（FNO 预测 0.1λ 散射场平面 → PWS 传播到任意 z → 加回背景场得总场，并可与仿真体数据对比）。

```
贴片形状 (SDF+Mask) ──FNO──► 近场截面场 (z=0.1λ)
                                   │
                                   ▼ PWS 物理传播
                            全空间场分布 E(x,y,z)
```

---

## 任务描述

给定 RIS 单元贴片形状（SDF + 二值掩码）与归一化频率，预测截面的复电磁场。

- 输入：`(2, 64, 64)`（ch0=SDF，ch1=Mask）+ 归一化频率 `freq (1,)`
- 输出：`(12, 64, 64)` = `[Ex_r, Ex_i, Ey_r, Ey_i, Ez_r, Ez_i, Hx_r, Hx_i, Hy_r, Hy_i, Hz_r, Hz_i]`
- 频率范围：8 ~ 12 GHz，步长 0.5 GHz，共 9 个频点
- 仿真：Floquet 端口垂直入射，周期边界条件

---

## 数据集

### 目录结构

```
data/data/<category>/<sample>/
    patch_plane.npy            (64, 64, 2)   ch0=SDF, ch1=Mask
    field_volume_merged.npy    (9, 22, 64, 64, 12)  全 z 层体数据（v2 格式）
    field_plane_d0.10.npy      (9, 64, 64, 12)      0.1λ 截面缓存（v2_cache 格式）
    *_S_Parameters.csv          S 参数
```

### 数据规模（实际扫描结果）

| 类别 | 样本数 | 频点数 | 记录数 |
|------|--------|--------|--------|
| 1_rectangle | 200 | 9 | 1800 |
| circle | 200 | 9 | 1800 |
| circle_ring | 200 | 9 | 1800 |
| cross | 200 | 9 | 1800 |
| square_ring | 200 | 9 | 1800 |
| **合计** | **1000** | **9** | **9000** |

### 数据格式分支（`data/dataset.py`）

- `v2_cache`：读 `field_plane_d0.10.npy`，按频点直接索引（无 z 相邻层）
- `v2`：读 `field_volume_merged.npy`，按 `d/λ` 线性插值取目标平面，
  同时返回 `out_zminus` / `out_zplus`（相邻 ±1mm 平面，供 PINN 中心差分）
- `v1`：读 `field_slice/*.npy`（仅电场，65×65）

### 数据划分

按形状分层划分（同一形状的所有频点在同一集合），训练 80% / 验证 20%，种子 42。

---

## 网络结构（当前代码）

### FNO2d（主模型，`patch2field/models/fno2d.py`）

```
输入 x (B, 2, 64, 64) + freq (B, 1)
    │
    ├── FourierFreqEncoder(freq) ──► freq_emb (B, 32)
    └── SDFGeomEncoder(x)        ──► geom_emb (B, 32)   # SDF 的低频 FFT 幅度特征
    │
    ▼ Lift: Conv2d(2→64, 1×1) + GELU
    ▼ pad +7px
    ├── FNOBlock × 5
    │      spectral = SpectralConv2d（频域卷积 + 超网络动态调制 Q/K 的交叉注意力）
    │      out = GELU(BN(spectral(x, freq_emb, geom_emb) + bypass(x)))
    │      x = x + out（残差）
    ▼ 去 padding
    ▼ shared_proj: Conv2d(64→64) + GELU
    ├── e_heads × 3：DirectionHead(64→2)      # Ex, Ey, Ez
    ├── 拼接 E 预测 → h_heads × 3：DirectionHead(64+6→2)  # Hx, Hy, Hz（自回归）
    ▼ cat
输出 (B, 12, 64, 64)
```

关键点：

- **SpectralConv2d 超网络注意力**：超网络（`hyper_wq` / `hyper_wk`）用条件向量
  动态生成 Q/K 投影增量 `ΔW_q(freq_emb)` / `ΔW_k(geom_emb)`，在频域做复数交叉注意力；
  `gamma` 初始 0.1 做残差门控。
- **方向独立输出头**：3 个 E 头 + 3 个 H 头（12 通道时），H 头以共享特征 + E 预测为输入。
- **6 通道模式**（`out_channels=6`）时退化为 3 个普通 `heads`，无 H。
- **shape_idx / type_idx 暂不参与计算**：模型接口已移除这两个参数；
  标签调制（双标签系统）当前未实现，文档不再声称具备该能力。

默认配置（`patch2field/config.py`）：`modes1=modes2=8, width=64, n_layers=5,
freq_dim=32, out_channels=12`，参数量约 **8.05M**。

### 对比模型（`patch2field/models/`）

| 模型 | 说明 |
|------|------|
| `fno2d_baseline.py` | 纯 FNO2d：无超网络/注意力，freq 展开为常数通道拼入输入（in=3） |
| `fno2d_film.py` | FNO + FiLM 条件调制 |
| `fno2d_lora.py` | FNO + LoRA 风格频率条件卷积核 |
| `unet2d.py` | 纯空间域 U-Net 基线 |

### 实验切换（`patch2field/experiment_config.py`）

改 `ACTIVE_EXPERIMENT` 即可切换 `fno_attention / fno_baseline / fno_baseline_large /
fno_baseline_parity(参数量对齐~8M) / fno_lora / fno_film / unet`，保存路径自动隔离到 `checkpoints/<name>/`。

---

## 损失函数（当前默认）

损失向量定义见 `patch2field/config.py`（L1~L9 编号）。默认启用：

| 编号 | 名称 | 权重 | 说明 |
|------|------|------|------|
| L3 | geom | 1.0 | 全场几何损失（方向角 + 相位 + 总场幅值） |
| L8 | ch_rel6_mean | 1.0 | 逐通道相对 L2 均值（与保存指标同口径） |
| L9 | pinn_maxwell | 0.0 | 麦克斯韦残差约束（需 v2 体数据提供相邻平面） |

注意：默认数据走 `v2_cache`（无相邻平面），故 L9 默认关闭；要启用 PINN
需使用 `v2`（体数据，`field_volume_merged.npy`）。若在 `v2_cache` 下误开启 L9，
训练启动时会打印警告。

输出标准化：训练集统计 12 通道统计量，训练用标准化空间，保存/评估自动反标准化。
默认 `OUTPUT_NORM_MODE="log"`（推荐）：对每个通道做 `sign(x)*log1p(|x|/scale)`
对数压缩动态范围后再逐通道 z-score，scale=训练集该通道 std。理由：电场
（std~1e2~1e3）与磁场（Hx~3，Hy/Hz~1）尺度相差约 3 个数量级，线性 z-score
会让小尺度通道的细微结构被大尺度通道淹没。切换 `OUTPUT_NORM_MODE="zscore"`
可恢复旧的逐通道线性标准化；checkpoint 中保存模式与参数，旧 checkpoint 自动按 zscore 解释。

---

## 训练配置（`config.py` 当前值）

| 参数 | 值 |
|------|----|
| 优化器 | AdamW |
| 学习率 | 1e-3（warmup 10 + cosine） |
| Weight Decay | 1e-4 |
| Batch Size | 16 |
| Epochs | 1000 |
| 早停 Patience | 200（按 val Geom） |
| Grad Clip | 1.0 |

## 保存规范（checkpoint）

`train.py` 保存的 `.pth` 包含：`epoch / model_state_dict / optimizer_state_dict /
val_loss / val_rel / val_geom / use_output_standardization`，标准化开启时含
`output_norm_mode` 与对应参数（zscore: mean/std；log: scale/logmean/logstd）。
`val_rel / val_geom` 为**物理空间**指标（先反归一化再计算），避免 log 归一化空间
导致方向角/相位失真；训练损失仍在归一化空间计算。

`output_norm_stats.npz` 保存于各实验目录（含模式与参数），便于离线评估反标准化。
训练启动时若 `SAVE_DIR` 不存在会自动创建（`train.py` 内置 `makedirs`）。

### ⚠️ checkpoint 说明

历史 checkpoint（旧架构、旧归一化）已清理删除；`checkpoints/` 现在为空，
下一次训练会自动重建目录。所有历史对比结果不作数，需用当前代码重新训练后再评估。

---

## 评估

- `patch2field/eval_generalization.py`：circle/circle_ring/cross/square_ring 训练 →
  rectangle（`1_rectangle`）泛化对比（主模型 vs 两个基线：`fno_baseline_large` 与
  参数量对齐的 `fno_baseline_parity`）；模型结构从 checkpoint 自动推断。
- `patch2field/ui_eval.py`：交互式评估 UI（误差热力图、样本浏览）。
- `field2field/`：PWS 传播工具（见其 README），把 FNO 预测截面传播到全空间。

## 文件说明（当前）

| 路径 | 说明 |
|------|------|
| `patch2field/train.py` | 主训练（v2/v2_cache 全频点） |
| `patch2field/config.py` | 全局配置（损失向量、模型、训练超参） |
| `patch2field/experiment_config.py` | 实验总控（切换模型与保存路径） |
| `patch2field/models/` | FNO 模型族（fno2d / baseline / film / lora / unet） |
| `patch2field/models/losses.py` | 损失函数（geom / DC-AC / 频域 / PINN） |
| `data/dataset.py` | 主数据集（v2_cache / v2 / v1） |
| `field2field/` | PWS 传播工具（.fld 读写、传播、可视化、FNO 到全空间衔接） |
| `LOSS_DESIGN.md` | 自适应权重损失设计（计划中，未实现） |
| `requirements.txt` | 依赖清单 |
