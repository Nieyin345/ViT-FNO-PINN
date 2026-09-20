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
