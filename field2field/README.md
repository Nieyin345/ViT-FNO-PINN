# HFSS 散射场 PWS 传播工具

从单截面场分布计算其他位置的场分布。

## 原理

```
散射场 = 总场 - base场（无贴片仿真）
预测总场 = PWS传播(散射场) + base场
```

散射场是贴片的二次辐射，含 +z 和 -z 两个方向分量，用 E+H 联合双向 PWS 传播。

## 文件说明

- `propagate_scatter.py` — 主入口
- `propagation.py` — PWS 传播核心（单向 / E+H 双向）
- `field_io.py` — `.fld` 文件读写
- `metrics_utils.py` — 相对 L2 误差
- `constants.py` — 物理常量

## 输入文件

| 文件 | 说明 |
|------|------|
| `e_field.fld` | 总场 E（有贴片） |
| `h_field.fld` | 总场 H（有贴片） |
| `e_base.fld`  | base场 E（无贴片） |
| `h_base.fld`  | base场 H（无贴片） |

## 用法

```bash
python propagate_scatter.py \
  --source-z-mm 3 \
  --targets-mm 5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43 \
  --freq-ghz 12 \
  --out-dir out_result
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source-z-mm` | 3.0 | 源截面 z 坐标 (mm) |
| `--targets-mm` | 5,7,...,43 | 目标截面列表 (mm) |
| `--freq-ghz` | 12.0 | 频率 (GHz) |
| `--n-medium` | 1.0 | 传播介质折射率 |
| `--out-dir` | out_result | 输出目录 |

## 注意事项

- 总场和 base场必须在同一频率点、同一网格导出
- HFSS 导出时选 **Complex** 格式
- 源截面建议选离贴片 2~3mm 以上，倏逝波衰减后精度更高
- 目标截面只能在源截面的 +z 方向（不能反传）
