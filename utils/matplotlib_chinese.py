#!/usr/bin/env python3
"""
matplotlib中文字体配置模块
在任何需要显示中文的脚本开头导入即可
"""

import matplotlib.pyplot as plt
import platform

def setup_chinese_font():
    """
    自动配置matplotlib中文字体
    支持Windows、macOS、Linux系统
    """
    system = platform.system()
    
    if system == "Windows":
        # Windows系统字体
        fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
    elif system == "Darwin":  # macOS
        # macOS系统字体
        fonts = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        # Linux系统字体
        fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = fonts
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 设置其他中文相关参数
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9

# 自动执行配置
setup_chinese_font()

# 提供一个测试函数
def test_chinese_display():
    """测试中文显示效果"""
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    
    ax.plot(x, y, 'b-', linewidth=2, label='正弦波')
    ax.set_title('中文字体测试 - 数学函数图像')
    ax.set_xlabel('角度 (弧度)')
    ax.set_ylabel('幅值')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加中文注释
    ax.annotate('最大值点', xy=(np.pi/2, 1), xytext=(np.pi/2, 1.2),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('chinese_font_test_module.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ 中文字体测试完成，图片已保存为 chinese_font_test_module.png")

if __name__ == '__main__':
    print("🎨 matplotlib中文字体配置模块")
    print("=" * 40)
    print("✅ 字体配置完成")
    print("\\n使用方法:")
    print("  import matplotlib_chinese  # 自动配置中文字体")
    print("  # 然后正常使用matplotlib即可显示中文")
    print("\\n或者:")
    print("  from matplotlib_chinese import setup_chinese_font")
    print("  setup_chinese_font()")
    print("\\n测试中文显示:")
    test_chinese_display()