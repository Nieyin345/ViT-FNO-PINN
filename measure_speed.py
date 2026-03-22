"""
推理速度测量 — 对比 FDFD vs ViT-FNO-PINN
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import time

import ceviche
from ceviche import fdfd_ez

import config as C
from models import FNO2d

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT   = f"{C.SAVE_DIR}/best.pth"

GRID_SIZE  = 64
C_0        = 3e8
WAVELENGTH = 1e-6
FREQ       = C_0 / WAVELENGTH
OMEGA      = 2 * np.pi * FREQ
PHYS_SIZE  = 4 * WAVELENGTH
DL         = PHYS_SIZE / GRID_SIZE
N_PML      = 10
SOURCE_IX  = GRID_SIZE // 4
SOURCE_IY  = GRID_SIZE // 2


def measure_fdfd(n_runs=10):
    eps = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.complex128)
    eps[20:44, 20:44] = 4.0
    eps_t = eps.T
    src = np.zeros((GRID_SIZE, GRID_SIZE), dtype=complex)
    src[SOURCE_IX, SOURCE_IY] = 1.0

    # warmup
    sim = fdfd_ez(OMEGA, DL, eps_t, [N_PML, N_PML])
    sim.solve(src)

    times = []
    for _ in range(n_runs):
        t0 = time.time()
        sim = fdfd_ez(OMEGA, DL, eps_t, [N_PML, N_PML])
        sim.solve(src)
        times.append(time.time() - t0)
    return np.mean(times) * 1000  # ms


def measure_model(model, device, n_runs=200):
    dummy_src = torch.zeros(1, 1, GRID_SIZE, GRID_SIZE, device=device)
    dummy_eps = torch.zeros(1, 1, GRID_SIZE, GRID_SIZE, device=device)
    inp = torch.cat([dummy_src, dummy_eps], dim=1)

    model.eval()
    # warmup
    for _ in range(10):
        with torch.no_grad():
            model(inp)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            model(inp)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / n_runs * 1000  # ms


def main():
    print("=" * 50)
    print("推理速度测量")
    print("=" * 50)

    # FDFD
    print("\n[1] ceviche FDFD ...")
    fdfd_ms = measure_fdfd(n_runs=5)
    print(f"    FDFD 平均耗时: {fdfd_ms:.1f} ms/sample")

    # 加载模型
    model = FNO2d(
        modes1=C.MODES1, modes2=C.MODES2, width=C.WIDTH, n_layers=C.N_LAYERS,
        out_channels=2, use_geo_encoder=C.USE_GEO_ENCODER,
        geo_embed_dim=C.GEO_EMBED_DIM, geo_depth=C.GEO_DEPTH,
        patch_size=C.PATCH_SIZE,
    )
    if os.path.exists(CKPT):
        model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
        print(f"\n模型加载: {CKPT}")
    else:
        print(f"\n警告: 未找到 {CKPT}，使用随机权重测速")

    # CPU 推理
    print("\n[2] ViT-FNO-PINN (CPU) ...")
    model_cpu = model.cpu()
    cpu_ms = measure_model(model_cpu, torch.device("cpu"), n_runs=100)
    print(f"    CPU 推理: {cpu_ms:.2f} ms/sample")

    # GPU 推理（如果有）
    gpu_ms = None
    if torch.cuda.is_available():
        print("\n[3] ViT-FNO-PINN (GPU) ...")
        model_gpu = model.cuda()
        gpu_ms = measure_model(model_gpu, torch.device("cuda"), n_runs=500)
        print(f"    GPU 推理: {gpu_ms:.2f} ms/sample")

    print("\n" + "=" * 50)
    print("汇总")
    print("=" * 50)
    print(f"  ceviche FDFD:          {fdfd_ms:8.1f} ms")
    print(f"  ViT-FNO-PINN (CPU):    {cpu_ms:8.2f} ms")
    if gpu_ms is not None:
        print(f"  ViT-FNO-PINN (GPU):    {gpu_ms:8.2f} ms")
        print(f"  加速比 (vs FDFD, GPU): {fdfd_ms/gpu_ms:8.0f}×")
    print(f"  加速比 (vs FDFD, CPU): {fdfd_ms/cpu_ms:8.0f}×")
    print("=" * 50)

    # 保存结果
    with open(f"{C.SAVE_DIR}/speed_results.txt", "w", encoding="utf-8") as f:
        f.write(f"FDFD: {fdfd_ms:.1f} ms\n")
        f.write(f"CPU:  {cpu_ms:.2f} ms\n")
        if gpu_ms:
            f.write(f"GPU:  {gpu_ms:.2f} ms\n")
    print(f"\n结果已保存: {C.SAVE_DIR}/speed_results.txt")


if __name__ == "__main__":
    main()
