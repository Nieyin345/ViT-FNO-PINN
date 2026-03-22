# 项目技术笔记：Physics-Informed FNO 电磁场预测

## 问题定义

用神经网络预测 2D TM 模式点源散射的电磁场（Ez 分量），输入为介质分布和点源位置，输出为复数 Ez 场（实部+虚部）。

控制方程为复数亥姆霍兹方程（ceviche 约定）：

```
∇²Ez + k²·ε_r·Ez = S_eff
```

其中 S_eff 是复数纯虚数点源，每个样本不同，需从 FDFD 真值反推。

---

## 数据生成

### 物理参数

| 参数 | 值 |
|------|-----|
| 波长 λ | 1 μm |
| 频率 f | 3×10¹⁴ Hz |
| 物理域尺寸 | 4λ × 4λ = 4 μm × 4 μm |
| 格点数 | 64 × 64 |
| 格点间距 dl | 62.5 nm = λ/16 |
| PML 宽度 | 10 格点 |
| 介质介电常数 | ε_r = 4（背景 ε_r = 1） |
| 点源位置 | ix=16, iy=32（域左侧 1/4 处） |

### 几何采样

每个样本随机生成一个正方形介质：
- 中心位置 cx, cy ∈ [0.35, 0.75]（归一化，避开点源区域）
- 边长 size ∈ [0.15, 0.45]（相对域尺寸）

### FDFD 求解（ceviche）

使用 `ceviche.fdfd_ez` 求解 TM 模式。注意 ceviche 坐标约定：
- eps_r 输入需转置：`eps_ceviche = eps_map.T`（ceviche 用 (Nx, Ny) 顺序）
- 输出 Ez 同样需转置回 (H, W)
- 点源在 ceviche 坐标中是 `src[SOURCE_IX, SOURCE_IY] = 1.0`

### 归一化与 s_eff 反推

Ez 场除以最大幅值归一化到 [-1, 1]，同时保存原始幅值 `scales.npy`。

等效源强度 s_eff 从归一化后的场反推：
```python
R_no_src = ∇²Ez + k_nd² · ε_r · Ez   # 不含源项的残差
s_eff = -R_no_src[SOURCE_IY, SOURCE_IX]  # 源点处残差即为 -s_eff
```

其中无量纲波数 `k_nd = 2π × (4/64)`（4λ 域，64 格点）。

s_eff 是复数纯虚数，|s_eff| ≈ 2.0，每个样本略有不同。

### 数据文件

```
data/em_data_meep/
  inputs.npy    (N, 2, 64, 64)  — ch0: source one-hot, ch1: permittivity [0,1]
  outputs.npy   (N, 2, 64, 64)  — ch0: Ez real, ch1: Ez imag，归一化到 [-1,1]
  scales.npy    (N,)             — 每个样本的归一化系数（原始最大幅值）
  s_effs.npy    (N, 2)           — 等效源强度 [real, imag]
  metadata.npy  (N, 3)           — 几何参数 [cx, cy, size]
```

---

## 网络结构：FNO2d

### 整体架构

```
输入 (B, 2, H, W)
  ↓ lift: Conv2d(2 → 64, 1×1)
  ↓ FNOBlock × 4
  ↓ proj: Conv2d(64 → 2, 1×1)  ← 线性投影，无激活
输出 (B, 2, H, W)  — [Ez_real, Ez_imag]
```

### FNOBlock

每个 FNOBlock = 谱卷积 + 旁路 1×1 卷积 + InstanceNorm + GELU：

```
x → SpectralConv2d(x) + Conv2d_1x1(x) → InstanceNorm → GELU → 输出
```

### SpectralConv2d

在频域对低频模态做线性混合：
1. `rfft2(x)` → 频域表示 (B, C, H, W//2+1)
2. 只保留左上角 (modes1, modes2) 和左下角 (modes1, modes2) 的低频分量
3. 用复数权重矩阵做 einsum 混合
4. `irfft2` 变回空间域

两组权重 w1（左上角正频率）、w2（左下角负频率对称部分）。

### 关键设计决策

**输出投影用线性层（无激活）**：最后的 proj 层只用单层 `Conv2d(64→2, 1×1)`，不加 GELU。原因：非线性激活会破坏 FNO 谱卷积产生的平滑周期性结构，导致预测场出现"破碎感"（波纹不连续）。

### 超参数

| 参数 | 值 |
|------|-----|
| modes1, modes2 | 16 |
| width | 64 |
| n_layers | 4 |
| in_channels | 2（source + permittivity） |
| out_channels | 2（Ez real + Ez imag） |
| 总参数量 | ~2.4M |

---

## 损失函数设计

### 总损失

```
L_total = λ_pde · L_pde + λ_bc · L_bc + λ_src · L_src
        + λ_nonzero · L_nonzero + λ_amp · L_amp
        + λ_data · L_data  (仅有监督样本)
```

### 各项说明

**L_pde — 亥姆霍兹方程残差**

复数形式，只在内部区域（排除 PML）计算：
```
R = ∇²Ez + k² · ε_r · Ez - S_eff
L_pde = mean(|R[p:-p, p:-p]|²)
```
- 拉普拉斯用中心差分，边界零填充
- Ez 拆成实部虚部分别 pad 再合并为复数
- S_eff 是 per-sample 复数点源，从 s_effs.npy 读取

**L_bc — PML 边界条件**

PML 四边（宽度 p=10）的 Ez 应趋近于零：
```
L_bc = mean(Ez[:, :, :p, :]² + Ez[:, :, -p:, :]² + ...)
```

**L_src — 源点非零约束**

防止模型输出零场（trivial solution），在源点位置用 log barrier：
```
L_src = mean(1 / (|Ez|_src + 1e-6))
```

**L_nonzero — 全场非零约束**

全场幅值的 log barrier，防止整体零解：
```
L_nonzero = mean(1 / (|Ez| + 1e-6))
```

**L_amp — 幅值上界约束**

防止无监督样本幅值爆炸（之前出现过 pred_max=22-28 的问题）：
```
amp_max = per-sample 最大幅值
L_amp = mean(clamp(amp_max - 1.5, min=0)²)
```
惩罚超过 1.5 的部分（归一化场最大值应为 1）。

**L_data — 有监督 MSE**

per-sample 归一化 MSE，消除不同样本幅值差异的影响：
```
scale = max(|Ez_true|)  per-sample
L_data = mean((Ez_pred/scale - Ez_true/scale)²)
```

### 损失权重

| 权重 | 值 | 说明 |
|------|-----|------|
| λ_pde | 3.0 | 物理约束，较大以约束无监督样本 |
| λ_bc | 0.1 | 边界条件 |
| λ_src | 1.0 | 源点非零 |
| λ_nonzero | 0.001 | 全场非零（很小，只防零解） |
| λ_amp | 1.0 | 幅值上界（硬编码在 PINNLoss 里） |
| λ_data | 200.0 | 有监督 MSE，主导梯度方向 |

---

## 训练策略

### 数据划分

- 总样本：200 个（前 200 个）
- 有监督：前 40 个（有真值标签）
- 无监督训练：40~200 中的 80%
- 验证集：40~200 中的 20%（约 32 个，有真值但训练时不用）

### 联合训练（Joint Training）

每个 epoch，对 train_loader（有监督 + 无监督混合）做一遍：

```python
for batch in train_loader:
    Ez = model(source, eps)
    
    # 全部样本：物理约束
    losses = pde_loss + bc_loss + src_loss + nonzero_loss + amp_loss
    
    # 有监督样本：额外加 data loss
    if supervised.any():
        losses += λ_data * data_loss(Ez[supervised], Ez_true[supervised])
    
    losses.backward()
```

验证集用真值 data_loss 评估，`best.pth` 保存验证集 data_loss 最小的模型。

### 优化器配置

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| 学习率 | 5e-4 |
| weight_decay | 1e-4 |
| 梯度裁剪 | max_norm=1.0 |
| 调度器 | CosineAnnealingLR |
| T_max | 500 epochs |
| eta_min | 1e-5 |

---

## 踩过的坑

### 1. ceviche 坐标约定

ceviche 用 (Nx, Ny) = (列, 行) 顺序，与 numpy 的 (行, 列) 相反。eps_r 和 src 都需要转置，Ez 输出也需要转置回来。

### 2. 亥姆霍兹方程符号

ceviche 的方程约定是 `∇²Ez + k²ε_r Ez = S_eff`，S_eff 是复数纯虚数。反推 s_eff 时用 `s_eff = -R_no_src[src_point]`。验证时排除源点（源点处有点源激励，残差不为零是正常的），内部区域相对残差均值约 0.001。

### 3. 无监督样本幅值爆炸

早期训练时，无监督样本的预测幅值高达 22-28（真值为 1）。原因是 λ_pde 太小（0.1），物理约束几乎不起作用。解决方案：提高 λ_pde 到 3.0，并加入 amplitude_loss 惩罚超过 1.5 的幅值。

### 4. 交替训练破坏相位

尝试过"先有监督纯 MSE，再无监督纯 PDE"的交替训练，结果 PDE 阶段把有监督阶段学到的相位覆盖掉（预测虚部符号反转）。原因：PDE loss 对无监督样本没有相位参考，可以收敛到任意相位的解。改回联合训练解决。

### 5. 输出层非线性破坏平滑性

原始 proj 层是 `Conv(64→32) + GELU + Conv(32→2)`，非线性激活破坏了 FNO 谱卷积的平滑周期性，导致预测场有"破碎感"（波纹不连续）。改为单层线性 `Conv(64→2)` 后改善。

### 6. 验证集指标不准

早期验证集用物理残差（pde+bc loss）评估，但物理残差不等于预测误差，导致 best.pth 保存的不是预测最准的模型。改为用真值 data_loss 评估验证集。

---

## 当前效果

| 指标 | 值 |
|------|-----|
| 训练集 rel-L2 | ~0.49 |
| 验证集 rel-L2 | ~0.76 |
| 有监督样本数 | 40 / 200 |

验证集预测场能看出波纹结构，但幅值偏低，泛化还有提升空间。
