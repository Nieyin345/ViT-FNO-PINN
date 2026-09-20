# 电磁超表面场预测模型文档

## 项目主线（不变）

1. **models/ + core/**：用 FNO（Fourier Neural Operator）从贴片几何（SDF + Mask）预测**近场截面**的场分布（12 通道：Ex/Ey/Ez + Hx/Hy/Hz 的实部虚部）。
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

### FNO2d（主模型，`models/fno2d.py`）

```
输入 x (B, 2, 64, 64) + freq (B, 1)
    │
    ├── FourierFreqEncoder(freq) ──► freq_emb (B, 32)
    └── SDFGeomEncoder(x)        ──► geom_emb (B, 32)   # SDF 的低频 FFT 谱特征（实部+虚部）
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

默认配置（`configs/base.py`）：`modes1=modes2=8, width=64, n_layers=5,
freq_dim=32, out_channels=12`，参数量约 **8.05M**。

### 对比模型（`models/`）

| 模型 | 说明 |
|------|------|
| `fno2d_baseline.py` | 纯 FNO2d：无超网络/注意力，freq 展开为常数通道拼入输入（in=3） |
| `fno2d_film.py` | FNO + FiLM 条件调制 |
| `fno2d_lora.py` | FNO + LoRA 风格频率条件卷积核 |
| `unet2d.py` | 纯空间域 U-Net 基线 |

### 实验切换（`configs/experiments.py`）

改 `ACTIVE_EXPERIMENT` 即可切换 `fno_attention / fno_baseline / fno_baseline_large /
fno_baseline_parity(参数量对齐~8M) / fno_lora / fno_film / unet`，保存路径自动隔离到 `checkpoints/<name>/`。

---

## 损失函数（当前默认）

损失向量定义见 `configs/base.py`（L1~L9 编号）。默认启用：

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

## 训练配置（`configs/base.py` 当前值）

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

`core/checkpoint.py` 是全仓库唯一的存取路径。`tools/train.py` 保存的 `.pth` 包含：

```
format_version     2
model_name         "fno_v2" 等注册名
model_config       ModelConfig.to_dict()   ← 重建模型所需的全部结构参数
model_state_dict
optimizer_state_dict
epoch / val_loss / val_rel / val_geom
use_output_standardization
normalizer         {output_norm_mode, ...}  ← 归一化模式与参数（嵌套）
```

加载走 `core.checkpoint.load(path)`：按 `model_config` 重建模型 → **`strict=True`**
载入权重 → 恢复 `OutputNormalizer`。**架构不符直接抛错**，不再有"匹配 N 层跳过
M 层"的静默降级路径。

`val_rel / val_geom` 为**物理空间**指标（先反归一化再计算），避免 log 归一化空间
导致方向角/相位失真；训练损失仍在归一化空间计算。

`output_norm_stats.npz` 保存于各实验目录（含模式与参数），便于离线评估反标准化。
训练启动时若保存目录不存在会自动创建。

### 指标口径有两套（跨文件比较时认准）

| 出处 | Geom | RelL2 |
|---|---|---|
| `tools/train.py`（写入 `summary.json`，也是 `best.pth` 的判据） | E(前6)/H(后6) 分组平均 | `channel_group_balanced_l2`（分组归一后平均） |
| `tools/eval.py` | 12 通道整体 | 12 通道整体展平的相对 L2 |

两者不可直接比大小。

### ⚠️ checkpoint 说明

**本次结构重构（统一基座）后，旧格式 checkpoint 全部作废**：旧 ckpt 没有
`model_name` / `model_config`，`core.checkpoint.load()` 会明确报错要求重训。
模型数学未变（同配置参数量与 state_dict 键名已逐项比对一致），但接口与键名有
调整，需用当前代码重新训练后再评估。

---

## 评估

- `tools/eval.py`：按 `--checkpoint` 评估一个或多个权重并横向汇总；模型结构
  由 checkpoint 自身记录的 `model_config` 决定，不再从权重形状反推。
- `tools/ui_eval.py`：交互式评估 UI（误差热力图、样本浏览）。
- `tools/compare.py`：对比实验矩阵 + 汇总表（`runs/summary.csv`）。
- `tools/smoke.py`：结构一致性冒烟测试（注册表 / 前向反向 / 配置往返 /
  `strict=True` 往返 / 架构错配必须报错 / 全部实验可解析），CPU 可跑。
- `field2field/`：PWS 传播工具（见其 README），把 FNO 预测截面传播到全空间。

## 文件说明（当前）

| 路径 | 说明 |
|------|------|
| `tools/train.py` | 主训练（argparse 驱动，v2/v2_cache 全频点） |
| `configs/base.py` | 全局配置（损失向量、训练超参、模型结构默认值） |
| `configs/experiments.py` | 实验总控（切换模型、超参与保存路径） |
| `core/base.py` | `FNOBase` 统一基座 + `Cond` 条件载体 |
| `core/config.py` | `ModelConfig` / `SpectralConfig`（结构参数唯一来源） |
| `core/registry.py` | 唯一的"模型名 → 类"注册表 + `build_model()` |
| `core/checkpoint.py` | 唯一的 checkpoint 存取（`strict=True`） |
| `core/encoders.py` | 频率 / 几何 / 方向编码器 |
| `core/spectral.py` | 频域卷积层（含超网络注意力，可配置） |
| `models/` | FNO 模型族（fno2d / v2 / baseline / film / lora / unet） |
| `models/losses.py` | 损失函数（geom / DC-AC / 频域 / PINN） |
| `data/dataset.py` | 主数据集（v2_cache / v2 / v1） |
| `field2field/` | PWS 传播工具（.fld 读写、传播、可视化、FNO 到全空间衔接） |
| `LOSS_DESIGN.md` | 自适应权重损失设计（计划中，未实现） |
| `CLAUDE.md` | 开发约定（目录职责、加新模型的步骤、容易踩的坑） |
| `requirements.txt` | 依赖清单 |
