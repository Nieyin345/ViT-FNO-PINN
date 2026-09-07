# 电磁超表面场预测

预测 RIS 单元贴片的近场电磁场分布，并用物理传播重建全空间场。

## 工作流

```
贴片形状 (SDF + Mask)
    │  FNO（patch2field）
    ▼
近场截面场 (E+H, 12 通道 @ 0.1λ)
    │  PWS 物理传播（field2field）
    ▼
全空间场分布 E(x, y, z)
```

## 目录结构

```
├── patch2field/       # FNO 训练与评估（预测近场截面）
│   ├── config.py      # 全局配置（损失、模型、训练超参）
│   ├── experiment_config.py  # 实验总控（切换模型）
│   ├── train.py       # 主训练（v2/v2_cache 全频点）
│   ├── eval_generalization.py# 形状泛化评估
│   ├── ui_eval.py     # 交互式评估 UI
│   └── models/        # FNO 模型族 + 损失函数
├── field2field/       # PWS 传播工具（截面 → 全空间）
│   ├── propagation.py # 角谱法传播核心
│   ├── propagate_scatter.py  # 主入口
│   └── field_io.py    # HFSS .fld 文件读写
├── data/              # 数据加载器 + 原始数据（不入库）
├── checkpoints/       # 模型权重与结果（不入库）
├── MODEL_DOC.md       # 模型架构详细文档
└── LOSS_DESIGN.md     # 自适应损失设计（计划中）
```

## 数据

- 主数据 `data/data/`：5 类贴片 × 200 样本 × 9 频点 = 9000 条，64×64，12 通道（E+H）
- 旧数据 `data/em_data/`：circle / cross_base（65×65，6 通道电场），历史实验用
- 详细格式见 `MODEL_DOC.md`

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 切换实验（默认 fno_attention），训练
python patch2field/train.py

# 形状泛化评估（自动从 checkpoint 推断模型结构）
python patch2field/eval_generalization.py

# 交互式评估 UI
python patch2field/ui_eval.py

# 端到端：FNO 预测近场截面 → PWS 传播到全空间（可对比仿真体数据）
python field2field/propagate_fno_prediction.py --sample-dir data/data/circle/circle_1 --freq-ghz 12 --ckpt checkpoints/fno_attention/best.pth --targets-mm 5,7,9,11,13,15,17,19,21,23 --out-dir out_fno_prop

# PWS 传播：从 HFSS 截面场计算全空间（用法见 field2field/README.md）
python field2field/propagate_scatter.py --source-z-mm 3 --targets-mm 5,7,9 --freq-ghz 12
```

## 说明

- checkpoints 下的 `.pth`/`.npz` 与数据文件已被 `.gitignore` 排除，不入版本库
- 输出归一化默认 `OUTPUT_NORM_MODE="log"`：`sign(x)*log1p(|x|/scale)` 压缩动态范围后逐通道 z-score，解决电场（~1e3）与磁场（~0.7~3）尺度差约 3 个数量级的问题；恢复旧行为可设 `"zscore"`。统计量与模式保存在 checkpoint，训练/评估/UI 自动反归一化
- 训练类别默认 `circle/circle_ring/cross/square_ring`（rectangle 留作形状泛化评估，数据目录 `1_rectangle`），见 `patch2field/config.py`
- `checkpoints/` 已清空（历史权重已删除），下次训练会自动重建对应目录
- 详细架构、损失与配置见 `MODEL_DOC.md`


