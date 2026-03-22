# ViT-FNO-PINN 电磁场预测

基于 Transformer 几何编码器 + 傅里叶神经算子 + 物理约束的 2D TM 电磁场预测。

## 项目结构

```
├── config.py              # 所有超参数（改这一个文件就够）
├── train_pinn.py          # 训练入口
├── eval_pinn.py           # 评估 + 可视化
├── verify_helmholtz.py    # 验证数据是否满足亥姆霍兹方程
├── measure_speed.py       # 推理速度测试
├── ablation_train.py      # 消融实验
│
├── models/
│   ├── fno.py             # ViT-FNO 模型（GeometryEncoder + FNOBlock）
│   └── losses.py          # PINN 损失函数（亥姆霍兹残差 + 数据损失）
│
├── data/
│   ├── generate_data.py   # 数据生成（ceviche FDFD，支持10种形状）
│   ├── dataset.py         # EMDataset
│   ├── splits.py          # 分层 train/val split
│   └── em_data_meep/      # 数据目录（npy 文件）
│
└── checkpoints/           # 模型权重 + 训练曲线
```

## 快速开始

### 1. 安装依赖

```bash
pip install torch numpy matplotlib ceviche
```

### 2. 生成数据

```bash
# 生成 10 种形状各 500 个样本（共 5000 个）
python data/generate_data.py --n_samples 500 --shape all

# 也可以单独生成某种形状并追加
python data/generate_data.py --n_samples 500 --shape circle --append
```

支持的形状：`square` `circle` `cross` `ring` `open_ring` `rect_ring` `sector` `ellipse` `triangle` `pentagon`

### 3. 训练

```bash
python train_pinn.py
```

### 4. 评估

```bash
python eval_pinn.py
```

生成：
- `checkpoints/eval_preview.png` — 训练/验证样本对比
- `checkpoints/eval_error_dist.png` — 全验证集误差分布
- `checkpoints/eval_worst10.png` — 误差最大的 10 个样本
- `checkpoints/eval_good_complex.png` — 高复杂度低误差样本

## 关键配置（config.py）

```python
# 数据
N_TOTAL    = 5000   # 总样本数（自动按形状均匀截取）
GROUP_SIZE = 500    # 每种形状的原始样本数
VAL_RATIO  = 0.2    # 验证集比例

# 模型
MODES1 = MODES2 = 24   # 傅里叶模态数
WIDTH  = 32             # FNO 隐层宽度
N_LAYERS = 4

# 训练
N_EPOCHS = 300
LR       = 5e-4
EARLY_STOP_PATIENCE = 2

# 物理约束
USE_PDE    = True
LAMBDA_PDE = 1.0
USE_CURRICULUM  = True   # 课程学习：pde 权重从 0 爬升
CURRICULUM_T0   = 50     # 第几 epoch 开始引入 pde
CURRICULUM_RAMP = 50     # 爬升持续 epoch 数
```

## 模型架构

**输入**: `(B, 2, 64, 64)` — 点源图 + 介质分布图

**GeometryEncoder（ViT）**
- 8×8 patch 嵌入 → 3 层 Transformer → 全局池化
- 输出每层 FNO 的 FiLM 调制参数（scale + shift）

**FNO 主干**
- 4 个 FNOBlock：谱卷积（保留 24 个傅里叶模态）+ 旁路卷积 + FiLM 调制

**输出**: `(B, 2, 64, 64)` — Ez 实部 + 虚部

参数量：约 260 万（WIDTH=32）

## 在 Kaggle 上训练

```python
# Cell 1: 安装依赖
!pip install ceviche -q

# Cell 2: 上传代码后切换目录
import os; os.chdir('/kaggle/working/your-project')

# Cell 3: 训练
!python train_pinn.py

# Cell 4: 查看结果
from IPython.display import Image
Image('checkpoints/eval_preview.png')
```
