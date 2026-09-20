# -*- coding: utf-8 -*-
"""一致性冒烟测试：统一基座的关键不变量。

CPU 即可跑完（**不需要 GPU**），可在服务器上作为提交前的快速自检：

    python -m tools.smoke

覆盖：
  1. 注册表无重名；每个注册模型都能 build_model 出实例。
  2. 每个模型 forward(x, freq) 于 (2,2,64,64)/(2,2)，输出 (2,out,64,64)，且可反向传播。
  3. ModelConfig.to_dict -> from_dict 往返相等（含嵌套 SpectralConfig 与 extra）。
  4. 每个模型 save -> load 的 strict=True 往返成功，且输出逐元素一致。
  5. **架构不符必须报错**：把 A 模型的权重按 B 模型的结构加载，必须抛错，
     而不是"匹配 N 层跳过 M 层"。这是本轮重构要根治的核心症状。
  6. 每个 configs/experiments.py 预设都能解析出合法的 ModelConfig + 模型名。
"""

import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

sys.modules.setdefault("numpy._core", np.core)

import models  # noqa: F401  触发自注册
from core import checkpoint as CK
from core.config import ModelConfig, SpectralConfig
from core.registry import available, build_model

H = W = 64
FAILS = []


def check(label, fn):
    """跑一个检查项，捕获异常记为失败而不是中断整个套件。"""
    try:
        detail = fn()
        print(f"  [OK]   {label}" + (f"  —— {detail}" if detail else ""))
        return True
    except Exception as e:
        print(f"  [FAIL] {label}\n         {type(e).__name__}: {e}")
        FAILS.append(label)
        return False


def make_inputs(n=2):
    x = torch.randn(n, 2, H, W)
    freq = torch.rand(n, 2)
    freq[:, 0] = freq[:, 0] * 0.8 + 0.1          # 归一化频率落在 [0.1, 0.9]
    freq[:, 1] = torch.randint(0, 2, (n,)).float()  # 极化索引 0/1
    return x, freq


def small_cfg(**kw):
    base = dict(width=32, n_layers=2, modes1=8, modes2=8)
    base.update(kw)
    return ModelConfig(**base)


def check_registry():
    names = available()
    assert names, "注册表为空——models/__init__.py 是否漏了 import？"
    assert len(names) == len(set(names)), "注册名重复"
    built = []
    for n in names:
        m = build_model(n, small_cfg())
        assert getattr(m, "model_name", None) == n, f"{n}: model_name 属性未设置"
        built.append(n)
    return f"{len(built)} 个: {', '.join(built)}"


def check_forward_backward():
    x, freq = make_inputs()
    lines = []
    for name in available():
        m = build_model(name, small_cfg())
        out = m(x, freq)
        assert out.shape == (2, m.cfg.out_channels, H, W), \
            f"{name}: 输出 {tuple(out.shape)} != (2,{m.cfg.out_channels},{H},{W})"
        assert torch.isfinite(out).all(), f"{name}: 输出含 NaN/Inf"
        out.sum().backward()
        grads = [p.grad for p in m.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), \
            f"{name}: 没有任何参数收到梯度"
        lines.append(name)
    return f"{len(lines)} 个模型前向+反向通过"


def check_config_roundtrip():
    cfg = ModelConfig(
        width=48, n_layers=4, modes1=12, modes2=10, freq_dim=16,
        spectral=SpectralConfig(attention=False, decay=True, decay_k0=5.0, decay_order=3),
        extra={"my_custom_param": 7},
    )
    back = ModelConfig.from_dict(cfg.to_dict())
    assert back == cfg, f"往返不等:\n  {cfg}\n  {back}"
    # 未知键自动进 extra（保证旧 checkpoint / 新模型字段可往返）
    d = cfg.to_dict()
    d["some_future_field"] = 123
    assert ModelConfig.from_dict(d).extra.get("some_future_field") == 123
    # 非法值必须报错
    for bad in ({"head": "nope"}, {"kernel_cond": "nope"}, {"out_channels": 11}):
        try:
            ModelConfig(**{**dict(width=32, n_layers=1), **bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法配置 {bad} 没有报错")
    return "往返相等 / 未知键入 extra / 非法值报错"


def check_checkpoint_roundtrip(tmpdir):
    x, freq = make_inputs()
    done = []
    for name in available():
        cfg = small_cfg()
        m = build_model(name, cfg)
        p = os.path.join(tmpdir, f"{name}.pth")
        CK.save(p, m, cfg, epoch=7, val_rel=0.6, val_geom=0.5)
        lc = CK.load(p, quiet=True)
        assert lc.model_name == name, f"{name}: model_name 读回为 {lc.model_name}"
        assert lc.epoch == 7, f"{name}: epoch 读回为 {lc.epoch}"
        # load() 默认 .eval()；把原模型也切 eval 才能逐元素对比
        m.eval()
        with torch.no_grad():
            a, b = m(x, freq), lc.model(x, freq)
        assert torch.allclose(a, b, atol=1e-6), f"{name}: 往返输出不一致"
        done.append(name)
    return f"{len(done)} 个模型 strict=True 往返一致"


def check_strict_loading_rejects_mismatch(tmpdir):
    """核心回归：架构不符必须抛错。

    历史上 eval 从权重形状反推 in_channels，把 fno_v2 的 6 通道 lift 误读成 2，
    随后 strict=False 静默把 lift 随机初始化——架构错的运行看起来像成功的运行。
    """
    x, freq = make_inputs()
    cases = [("fno_v2", "fno_baseline"), ("fno_baseline", "fno_v2"),
             ("fno_attention", "fno_film")]

    ok = []
    for src, wrong in cases:
        src_cfg = small_cfg()
        m = build_model(src, src_cfg)
        p = os.path.join(tmpdir, f"mismatch_{src}.pth")
        CK.save(p, m, src_cfg, epoch=1)

        # 手工构造"用错误架构去加载"的 checkpoint
        import torch as _t
        ck = _t.load(p, map_location="cpu", weights_only=False)
        ck["model_name"] = wrong
        p_bad = os.path.join(tmpdir, f"bad_{src}_as_{wrong}.pth")
        _t.save(ck, p_bad)

        try:
            CK.load(p_bad, quiet=True)
        except Exception as e:
            ok.append(f"{src}→{wrong}: {type(e).__name__}")
        else:
            raise AssertionError(
                f"{src} 的权重被 {wrong} 的架构**静默加载**了——strict 检查失效！"
            )
    return "; ".join(ok)


def check_metric_independence():
    """**"损失可以随便改而结果仍可比"这条不变式的守卫。**

    保存/早停指标 val_geom / val_rel 必须**不读任何损失配置**，否则换一版损失后
    新旧数字就没有共同刻度，整个"基线冻结 + 主模型迭代损失"的流程不成立。

    实现上靠的是：Trainer.geom_metric 调用 field_geom_loss_eh(pred, true) 用子权重
    默认值 1/1/1，Trainer.rel_l2_error 调用 channel_group_balanced_l2(pred, true) 用
    h_boost 默认 1.0 —— 都不透传 config 里的训练侧旋钮。

    历史上 val_geom 一直是干净的，但这条性质**没有任何东西守着**：
    只要有人顺手把 FIELD_H_GROUP_WEIGHT 传进 geom_metric，指标就随损失漂移，
    而终端上完全看不出来。所以这里把它钉死。
    """
    import torch
    from configs import base as C
    from models.losses import field_geom_loss_eh, channel_group_balanced_l2

    g = torch.Generator().manual_seed(0)
    p = torch.randn(2, 12, 16, 16, generator=g)
    t = torch.randn(2, 12, 16, 16, generator=g)

    base_geom = field_geom_loss_eh(p, t).item()
    base_rel = channel_group_balanced_l2(p, t).item()

    # 把损失侧旋钮全部打到极端值
    saved = {k: getattr(C, k, None) for k in (
        "FIELD_H_GROUP_WEIGHT", "FIELD_DIR_ANGLE_WEIGHT",
        "FIELD_COMPLEX_PHASE_WEIGHT", "FIELD_TOTAL_MAG_WEIGHT",
        "LOSS_WEIGHT_VECTOR", "LOSS_ENABLE_VECTOR")}
    try:
        C.FIELD_H_GROUP_WEIGHT = 99.0
        C.FIELD_DIR_ANGLE_WEIGHT = 0.001
        C.FIELD_COMPLEX_PHASE_WEIGHT = 77.0
        C.FIELD_TOTAL_MAG_WEIGHT = 0.5
        C.LOSS_WEIGHT_VECTOR = [9.0] * 9

        after_geom = field_geom_loss_eh(p, t).item()
        after_rel = channel_group_balanced_l2(p, t).item()
    finally:
        for k, v in saved.items():
            if v is not None:
                setattr(C, k, v)

    assert abs(base_geom - after_geom) < 1e-9, (
        f"val_geom 随损失配置变化了（{base_geom} -> {after_geom}）——"
        f"指标不再与损失无关，换一版损失后新旧结果不可比"
    )
    assert abs(base_rel - after_rel) < 1e-9, (
        f"val_rel 随损失配置变化了（{base_rel} -> {after_rel}）"
    )

    # 静态确认调用点没有把 config 值传进去
    src = open(os.path.join(_ROOT, "tools", "train.py"), encoding="utf-8").read()
    gm = src.split("def geom_metric")[1].split("\n    def ")[0]
    rl = src.split("def rel_l2_error")[1].split("\n    def ")[0]
    assert "field_geom_loss_eh(pred_p, true_p)" in gm, \
        "geom_metric 似乎给 field_geom_loss_eh 传了额外参数（子权重会漂移）"
    assert "channel_group_balanced_l2(pred_p, true_p)" in rl, \
        "rel_l2_error 似乎给 channel_group_balanced_l2 传了 h_boost（会漂移）"

    return "val_geom / val_rel 与损失配置无关（打乱损失旋钮后逐位不变）"


def check_arch_fingerprint():
    """架构指纹必须是"改模型类才变、重训/换损失不变"。

    整个"基线冻结 + 主模型迭代损失"的流程都挂在这条上：
    指纹不变 → 旧 checkpoint 还加载得进去 → 旧结果仍然有效，不必重跑。

    它误报成"变了"的代价是白白重跑一代；误报成"没变"的代价是拿两个
    不可比的数字当可比——所以要两个方向都钉住。
    """
    from configs import base as C
    from configs import scenario as SC
    from core.config import ModelConfig
    from core.registry import build_model

    cfg = ModelConfig(width=32, n_layers=2, modes1=8, modes2=8)
    fp1 = SC.arch_fingerprint(build_model("fno_v2", cfg))
    fp2 = SC.arch_fingerprint(build_model("fno_v2", cfg))
    assert fp1 == fp2, "同一配置两次构造的指纹不同（权重随机初始化泄漏进指纹了？）"

    # 不同配置 → 不同指纹
    cfg_wide = ModelConfig(width=48, n_layers=2, modes1=8, modes2=8)
    assert SC.arch_fingerprint(build_model("fno_v2", cfg_wide)) != fp1, \
        "改了 width 指纹没变"
    cfg_deep = ModelConfig(width=32, n_layers=3, modes1=8, modes2=8)
    assert SC.arch_fingerprint(build_model("fno_v2", cfg_deep)) != fp1, \
        "改了层数指纹没变"

    # 从存档的 (model_name, model_config) 重算，应与训练时记的一致
    assert SC.arch_fingerprint_from_cfg("fno_v2", cfg.to_dict()) == fp1, \
        "从 model_config 重算的指纹与训练时不符——report 会误判全部 run 过期"

    # 模型已被删掉时必须抛错（report 据此标为已失效，而不是崩掉）
    try:
        SC.arch_fingerprint_from_cfg("fno_does_not_exist", cfg.to_dict())
    except Exception:
        pass
    else:
        raise AssertionError("未知模型名没有抛错——report 无法识别失效的 run")

    # 训练侧必须真的把它写进 summary / checkpoint
    src = open(os.path.join(_ROOT, "tools", "train.py"), encoding="utf-8").read()
    def body(fn):
        return src.split(f"def {fn}")[1].split("\n    def ")[0]
    assert "arch_fingerprint" in body("write_summary"), \
        "summary.json 没写 arch_fingerprint —— report 无法判断结果是否过期"
    assert "arch_fingerprint" in body("_save"), \
        "checkpoint 没写 arch_fingerprint"

    return "同配置稳定 / 改配置变化 / 可从存档重算 / 未知模型抛错"


def check_scenario_partition():
    """场景/损失两桶必须不重叠，且覆盖 base.py 里所有大写配置项。

    漏掉一个键 = 它既不进场景指纹也不进损失食谱，改了它不会有任何记录。
    """
    from configs import base as C
    from configs import scenario as SC

    s, l = set(SC.SCENARIO_KEYS), set(SC.LOSS_KEYS)
    overlap = s & l
    assert not overlap, f"场景与损失两桶重叠: {sorted(overlap)}"

    # 模型结构键两桶都不进（由 ModelConfig 记录），其余必须被覆盖
    model_keys = {
        "MODEL_NAME", "model", "SAVE_DIR", "MODES1", "MODES2", "WIDTH", "N_LAYERS",
        "FREQ_DIM", "PADDING", "COND_CHANNELS", "OUT_CHANNELS", "HEAD_MODE",
        "MODEL_EXTRA", "PROJ_DROPOUT", "SPECTRAL_DECAY", "SPECTRAL_DECAY_K0",
        "SPECTRAL_DECAY_ORDER", "SPECTRAL_ATTENTION_ENABLE",
        "SPECTRAL_FREQ_CONDITIONING", "SPECTRAL_GEOM_CONDITIONING",
        "SPECTRAL_GAMMA_INIT", "SPECTRAL_GAMMA_FIXED", "ACTIVE_EXPERIMENT",
        "KERNEL_COND_MODE", "USE_SINE_INPUT", "USE_POL_COND", "LORA_RANK",
        "SAVE_LATEST_EVERY", "ONLY_SAVE_PTH", "VAL_EVERY", "NUM_WORKERS",
        "PIN_MEMORY", "USE_SINGLE_OUTPUT_HEAD", "IN_CHANNELS",
    }
    allkeys = {k for k in vars(C) if k.isupper() and not k.startswith("_")}
    missing = allkeys - s - l - model_keys
    assert not missing, (
        f"这些配置项既不在 SCENARIO_KEYS 也不在 LOSS_KEYS，改了不会有记录: "
        f"{sorted(missing)}"
    )

    # 指纹必须真的随场景变化
    snap = SC.snapshot(SC.SCENARIO_KEYS)
    f1 = SC.fingerprint(snap)
    snap2 = dict(snap)
    snap2["OUTPUT_NORM_MODE"] = "zscore"
    assert SC.fingerprint(snap2) != f1, "改 OUTPUT_NORM_MODE 指纹没变"

    # 划分指纹：同样的索引必须得到同样的哈希，不同索引必须不同
    a = SC.split_fingerprint([1, 2, 3], [4, 5])
    b = SC.split_fingerprint([3, 2, 1], [5, 4])   # 顺序无关
    c = SC.split_fingerprint([1, 2, 3], [4, 6])
    assert a == b, "划分指纹对索引顺序敏感（应当无关）"
    assert a != c, "划分变了指纹没变"

    return (f"两桶不重叠、覆盖全部配置项；场景指纹随配置变化；"
            f"划分指纹顺序无关且能检出增删")


def check_experiments_resolve():
    from configs import base as C
    from configs.experiments import EXPERIMENTS
    seen = []
    for name in sorted(EXPERIMENTS):
        C.apply_experiment(name)
        mn = C.model_name_from_globals()
        assert mn in available(), f"{name}: 模型名 {mn} 未注册"
        cfg = C.model_config_from_globals()
        assert cfg.out_channels % 2 == 0
        seen.append(name)
    C.apply_experiment("fno_attention")   # 复位
    return f"{len(seen)} 个预设实验均可解析"


def main():
    print("=" * 68)
    print("统一基座冒烟测试")
    print("=" * 68)

    print("\n[1] 注册表")
    check("注册名唯一 + 均可构造", check_registry)

    print("\n[2] 前向 / 反向")
    check("forward(x, freq) 形状 + backward 梯度", check_forward_backward)

    print("\n[3] 配置往返")
    check("ModelConfig 序列化不变量", check_config_roundtrip)

    with tempfile.TemporaryDirectory(prefix="p2f_smoke_") as tmpdir:
        print("\n[4] checkpoint 往返（strict=True）")
        check("save -> load 输出一致", lambda: check_checkpoint_roundtrip(tmpdir))

        print("\n[5] 架构不符必须报错（核心回归）")
        check("错误架构被拒绝", lambda: check_strict_loading_rejects_mismatch(tmpdir))

    print("\n[6] 预设实验")
    check("experiments 全部可解析", check_experiments_resolve)

    print("\n[7] 指标与损失解耦（'损失可迭代、结果仍可比'的守卫）")
    check("val_geom / val_rel 不随损失配置变化", check_metric_independence)

    print("\n[8] 场景 / 损失 划分")
    check("两桶不重叠且覆盖全部配置项", check_scenario_partition)

    print("\n[9] 架构指纹（判断旧结果是否过期）")
    check("改模型类才变，重训/换损失不变", check_arch_fingerprint)

    print("\n" + "=" * 68)
    if FAILS:
        print(f"失败 {len(FAILS)} 项:")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
