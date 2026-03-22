"""
Windows 下直接运行，自动调用 WSL 中的 MEEP 生成数据
直接运行: python run_meep.py
"""

import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

def run_meep_in_wsl(num_samples=20):
    """在 WSL 中运行 MEEP 生成数据"""
    
    # 获取当前目录的 WSL 路径
    current_dir = os.getcwd().replace('\\', '/')
    drive = current_dir[0].lower()
    wsl_path = f"/mnt/{drive}/{current_dir[3:]}"
    
    print("=" * 60)
    print("使用 WSL + MEEP 生成电磁场数据")
    print("=" * 60)
    print(f"样本数: {num_samples}")
    print()
    
    # 构建命令
    cmd = f'source ~/miniconda3/etc/profile.d/conda.sh && conda activate meep && cd {wsl_path} && python generate_em_data_meep.py {num_samples}'
    
    try:
        print("正在 WSL 中运行 MEEP...")
        print("-" * 60)
        
        result = subprocess.run(
            ['wsl', 'bash', '-c', cmd],
            capture_output=False,
            text=True
        )
        
        print("-" * 60)
        
        if result.returncode == 0:
            print("\n✓ MEEP 数据生成成功！")
            return True
        else:
            print(f"\n✗ 失败，返回码: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n✗ 出错: {e}")
        return False


def visualize_data(data_dir='em_data_meep'):
    """可视化生成的数据"""
    
    if not os.path.exists(os.path.join(data_dir, 'inputs.npy')):
        print(f"数据目录 {data_dir} 不存在")
        return
    
    print(f"\n加载数据...")
    inputs = np.load(os.path.join(data_dir, 'inputs.npy'))
    outputs = np.load(os.path.join(data_dir, 'outputs.npy'))
    
    print(f"输入形状: {inputs.shape}")
    print(f"输出形状: {outputs.shape}")
    
    # 可视化
    num_samples = min(3, len(inputs))
    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        source_map = inputs[i, 0]
        epsilon_r = inputs[i, 1]
        E_real = outputs[i, 0]
        E_imag = outputs[i, 1]
        E_magnitude = np.sqrt(E_real**2 + E_imag**2)
        
        axes[i, 0].imshow(source_map, cmap='hot', origin='lower')
        axes[i, 0].set_title(f'Sample {i} - Source')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(epsilon_r, cmap='viridis', origin='lower')
        axes[i, 1].set_title('Permittivity')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(E_real, cmap='RdBu', origin='lower', vmin=-1, vmax=1)
        axes[i, 2].set_title('E Real')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(E_imag, cmap='RdBu', origin='lower', vmin=-1, vmax=1)
        axes[i, 3].set_title('E Imag')
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(E_magnitude, cmap='hot', origin='lower')
        axes[i, 4].set_title('E Magnitude')
        axes[i, 4].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(data_dir, 'visualization.png')
    plt.savefig(output_path, dpi=150)
    print(f"可视化已保存: {output_path}")
    plt.close()


if __name__ == '__main__':
    # 从命令行参数获取样本数，默认100
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    print(f"准备生成 {num_samples} 个样本...")
    print(f"预计用时: {num_samples * 0.28:.1f} 秒")
    
    # 生成数据
    success = run_meep_in_wsl(num_samples=num_samples)
    
    # 可视化
    if success:
        visualize_data('em_data_meep')
    
    print("\n完成！")
