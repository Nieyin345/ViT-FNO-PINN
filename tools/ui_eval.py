#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
现代化的tkinter评估界面 - 改进版
使用更好的样式和布局
"""

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import defaultdict
from torch.utils.data import Subset, ConcatDataset, DataLoader
from scipy.ndimage import distance_transform_edt

import models  # noqa: F401  触发所有模型自注册
from core.checkpoint import load as load_checkpoint
from core.registry import build_model, available as available_models
from configs import base as C
from data.dataset import EMMultiFreqDataset
from data.norm_utils import OutputNormalizer
from models.losses import field_geom_loss


def _build_model():
    """按 configs 里当前实验的模型名与结构参数构建模型。

    旧实现维护了一份本地 `_MODEL_REGISTRY`（5 个模型，缺 fno_v2），且未知名字用
    `_MODEL_REGISTRY.get(model_name, FNO2d)` **静默退回 FNO2d**，随后又用
    `strict=False` 把权重塞进去——于是一个名字写错的运行能一路跑到出图。
    现在走全仓库唯一注册表，未知名字直接抛错并列出可选值。

    注意：GUI 加载 checkpoint 时**以 checkpoint 为准**（见
    `_load_checkpoint_from_file`），本函数只在没有 checkpoint 时用于取随机模型。
    """
    name = C.model_name_from_globals()
    cfg = C.model_config_from_globals()
    model = build_model(name, cfg)
    print(f"[模型] {name}  参数量: {sum(p.numel() for p in model.parameters()):,}")
    return model

# 设置matplotlib样式
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


def _setup_matplotlib_cjk_font():
    """Use a CJK-capable font when available to avoid glyph-missing warnings."""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            break
    plt.rcParams['axes.unicode_minus'] = False


_setup_matplotlib_cjk_font()


def build_split(dataset):
    """按类别分层划分，保证同一形状不会跨训练/验证。"""
    category_groups = defaultdict(lambda: defaultdict(list))
    for i, s in enumerate(dataset.samples):
        category = s["category"]
        shape_name = s["name"]
        category_groups[category][shape_name].append(i)

    rng = torch.Generator().manual_seed(getattr(C, "SHUFFLE_SEED", 42))
    train_idx, val_idx = [], []

    for category in sorted(category_groups.keys()):
        shapes = sorted(category_groups[category].keys())
        n_shapes = len(shapes)
        n_val_shapes = max(1, int(n_shapes * getattr(C, "VAL_RATIO", 0.2)))

        perm = torch.randperm(n_shapes, generator=rng).tolist()
        val_shapes = set(shapes[i] for i in perm[:n_val_shapes])

        for shape_name, idxs in category_groups[category].items():
            if shape_name in val_shapes:
                val_idx.extend(idxs)
            else:
                train_idx.extend(idxs)

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def dc_ac_error(pred_np, true_np, ch):
    """计算 Ex/Ey/Ez 单物理分量(实部+虚部)相对 L2。"""
    t = true_np[ch:ch + 2]
    p = pred_np[ch:ch + 2]
    return np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-6)


def amp_l2_error(pred_np, true_np, ch):
    """计算 Ex/Ey/Ez 幅值图相对 L2（更接近肉眼对比）。"""
    t_amp = np.sqrt(true_np[ch] ** 2 + true_np[ch + 1] ** 2)
    p_amp = np.sqrt(pred_np[ch] ** 2 + pred_np[ch + 1] ** 2)
    return np.linalg.norm(p_amp - t_amp) / (np.linalg.norm(t_amp) + 1e-6)


def rel_l2_error_batch(pred: torch.Tensor, true: torch.Tensor) -> float:
    """按通道计算相对 L2，再取平均。"""
    B, nC = pred.shape[:2]
    total = 0.0
    for i in range(nC):
        diff = (pred[:, i] - true[:, i]).reshape(B, -1).norm(dim=1)
        norm = true[:, i].reshape(B, -1).norm(dim=1).clamp(min=1e-6)
        total += (diff / norm).mean()
    return (total / nC).item()

class EvalApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🔬 电磁场预测模型评估")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # 设置现代化样式
        self.setup_styles()
        
        self._model = None
        self._ckpt_meta = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._train_ds = None
        self._val_ds = None
        self._running = False
        self._warned_channel_mismatch = False
        self._output_normalizer = None
        
        self.build_ui()
        
    def setup_styles(self):
        """设置现代化样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置样式
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'), foreground='#2c3e50')
        style.configure('Heading.TLabel', font=('Arial', 10, 'bold'), foreground='#34495e')
        style.configure('Modern.TButton', font=('Arial', 9))
        style.configure('Success.TLabel', foreground='#27ae60', font=('Arial', 9, 'bold'))
        style.configure('Error.TLabel', foreground='#e74c3c', font=('Arial', 9, 'bold'))
        
        # 按钮颜色配置
        style.map('Modern.TButton',
                 background=[('active', '#3498db'), ('pressed', '#2980b9')])
    
    def build_ui(self):
        """构建现代化UI"""
        # 主容器
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        self.build_control_panel(main_container)
        
        # 右侧显示区域
        self.build_display_area(main_container)
        
    def build_control_panel(self, parent):
        """构建控制面板"""
        # 控制面板框架 - 添加滚动功能
        control_main_frame = ttk.Frame(parent)
        control_main_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 创建Canvas和Scrollbar用于滚动
        canvas = tk.Canvas(control_main_frame, width=280, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(control_main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 鼠标滚轮绑定
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind("<MouseWheel>", _on_mousewheel)
        scrollable_frame.bind("<MouseWheel>", _on_mousewheel)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 控制面板内容
        control_frame = ttk.LabelFrame(scrollable_frame, text="⚙️ 控制面板", padding=8)
        control_frame.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        
        # 标题
        title_label = ttk.Label(control_frame, text="模型评估工具", style='Title.TLabel')
        title_label.pack(pady=(0, 8))
        
        # 数据配置区域
        self.build_data_config(control_frame)
        
        # 分隔线
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=8)
        
        # 模型配置区域
        self.build_model_config(control_frame)
        
        # 分隔线
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=8)
        
        # 操作按钮区域
        self.build_action_buttons(control_frame)
        
        # 状态显示区域
        self.build_status_area(control_frame)
    
    def build_data_config(self, parent):
        """构建数据配置区域"""
        data_frame = ttk.LabelFrame(parent, text="📁 数据配置", padding=6)
        data_frame.pack(fill=tk.X, pady=(0, 6))
        
        # 数据目录
        ttk.Label(data_frame, text="数据目录:", style='Heading.TLabel').pack(anchor=tk.W)
        dir_frame = ttk.Frame(data_frame)
        dir_frame.pack(fill=tk.X, pady=(2, 6))
        
        self.data_var = tk.StringVar(value=os.path.abspath(os.path.join(_HERE, C.DATA_DIR)))
        ttk.Entry(dir_frame, textvariable=self.data_var, font=('Arial', 9)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(dir_frame, text="浏览", command=self.browse_data, style='Modern.TButton').pack(side=tk.RIGHT, padx=(3, 0))
        
        # 类别选择
        ttk.Label(data_frame, text="选择类别:", style='Heading.TLabel').pack(anchor=tk.W, pady=(6, 2))
        
        # 类别列表框架
        cat_frame = ttk.Frame(data_frame)
        cat_frame.pack(fill=tk.X)
        
        # 类别列表
        self.cat_listbox = tk.Listbox(cat_frame, selectmode=tk.MULTIPLE, height=3, font=('Arial', 8))
        scrollbar = ttk.Scrollbar(cat_frame, orient=tk.VERTICAL, command=self.cat_listbox.yview)
        self.cat_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.cat_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 类别选择按钮
        cat_btn_frame = ttk.Frame(data_frame)
        cat_btn_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Button(cat_btn_frame, text="全选", command=self.select_all_cats, style='Modern.TButton').pack(side=tk.LEFT)
        ttk.Button(cat_btn_frame, text="清空", command=self.deselect_all_cats, style='Modern.TButton').pack(side=tk.LEFT, padx=(3, 0))
        
        # 频率选择
        ttk.Label(data_frame, text="频率选择:", style='Heading.TLabel').pack(anchor=tk.W, pady=(6, 2))
        self.freq_var = tk.StringVar(value="All")
        freq_combo = ttk.Combobox(data_frame, textvariable=self.freq_var, state="readonly", font=('Arial', 9))
        freq_combo['values'] = ["All"] + [f"{f:.1f} GHz" for f in [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]]
        freq_combo.pack(fill=tk.X)

        # 评估数据集选择
        ttk.Label(data_frame, text="评估数据集:", style='Heading.TLabel').pack(anchor=tk.W, pady=(6, 2))
        self.split_var = tk.StringVar(value="验证集")
        split_combo = ttk.Combobox(data_frame, textvariable=self.split_var, state="readonly", font=('Arial', 9))
        split_combo['values'] = ["验证集", "训练集", "训练+验证"]
        split_combo.pack(fill=tk.X)
        
        # 刷新类别列表
        self.refresh_categories()
    
    def build_model_config(self, parent):
        """构建模型配置区域"""
        model_frame = ttk.LabelFrame(parent, text="🤖 模型配置", padding=6)
        model_frame.pack(fill=tk.X, pady=(0, 6))
        
        # 检查点路径
        ttk.Label(model_frame, text="模型检查点:", style='Heading.TLabel').pack(anchor=tk.W)
        ckpt_frame = ttk.Frame(model_frame)
        ckpt_frame.pack(fill=tk.X, pady=(2, 0))
        
        self.ckpt_var = tk.StringVar(value=os.path.abspath(os.path.join(_HERE, C.SAVE_DIR, "best.pth")))
        ttk.Entry(ckpt_frame, textvariable=self.ckpt_var, font=('Arial', 9)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(ckpt_frame, text="浏览", command=self.browse_checkpoint, style='Modern.TButton').pack(side=tk.RIGHT, padx=(3, 0))
    
    def build_action_buttons(self, parent):
        """构建操作按钮区域"""
        action_frame = ttk.LabelFrame(parent, text="🚀 操作", padding=6)
        action_frame.pack(fill=tk.X, pady=(0, 6))
        
        # 主要操作按钮
        buttons = [
            ("🔄 加载数据和模型", self.load_data_model),
            ("📊 计算误差分布", self.compute_error_distribution),
            ("🔥 类别-频率热力图", self.generate_heatmap),
            ("🌡️ 全集场强热力图", self.generate_field_stats_heatmap),
            ("👁️ 可视化预测结果", self.visualize_predictions),
            ("🔍 数据诊断", self.diagnose_data),
        ]
        
        for text, command in buttons:
            btn = ttk.Button(action_frame, text=text, command=command, style='Modern.TButton')
            btn.pack(fill=tk.X, pady=1)
    
    def build_status_area(self, parent):
        """构建状态显示区域"""
        status_frame = ttk.LabelFrame(parent, text="📋 状态", padding=6)
        status_frame.pack(fill=tk.X, pady=(0, 6))
        
        # 状态文本区域 - 使用Text widget以支持多行和滚动
        status_text_frame = ttk.Frame(status_frame)
        status_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.status_text = tk.Text(status_text_frame, height=6, width=32, wrap=tk.WORD, 
                                  font=('Arial', 8), bg='#ffffff', fg='#333333')
        status_scrollbar = ttk.Scrollbar(status_text_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 清空日志按钮
        clear_btn_frame = ttk.Frame(status_frame)
        clear_btn_frame.pack(fill=tk.X, pady=(3, 0))
        ttk.Button(clear_btn_frame, text="清空日志", command=self.clear_status_log, 
                  style='Modern.TButton').pack(side=tk.RIGHT)
        
        # 初始状态
        self.status_text.insert(tk.END, "✅ 就绪\n等待加载数据和模型...")
        self.status_text.configure(state=tk.DISABLED)
        
        # 进度条
        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill=tk.X, pady=(3, 0))
        
        ttk.Label(progress_frame, text="进度:", font=('Arial', 8, 'bold')).pack(anchor=tk.W)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, length=260)
        self.progress_bar.pack(fill=tk.X, pady=(1, 0))
    
    def build_display_area(self, parent):
        """构建显示区域"""
        display_frame = ttk.LabelFrame(parent, text="📈 结果显示", padding=10)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建notebook用于多标签页
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 创建各个标签页
        self.create_tabs()
    
    def create_tabs(self):
        """创建标签页"""
        # 误差分布标签页
        self.error_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.error_frame, text="📊 误差分布")
        
        # 热力图标签页
        self.heatmap_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.heatmap_frame, text="🔥 误差热力图")
        
        # 场强均值标签页
        self.field_stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.field_stats_frame, text="🌡️ 场强热力图")
        
        # 可视化标签页（幅值）
        self.viz_amp_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_amp_frame, text="👁️ 幅值误差")

        # 可视化标签页（复数实部/虚部）
        self.viz_complex_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_complex_frame, text="🧮 复数误差")
        
        # 统计信息标签页
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="📋 统计信息")
        
        # 自由手绘预测标签页
        self.draw_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.draw_frame, text="✍️ 自由绘制预测")
        
        # 构建统计和绘制内容
        self.build_stats_content()
        self.build_draw_content()
        
    def build_draw_content(self):
        """构建自由绘制预测的界面内容"""
        # 左侧绘制区
        left_panel = ttk.Frame(self.draw_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        ttk.Label(left_panel, text="✨ 在黑板上画出贴片形状 (白=金属):", font=('微软雅黑', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(left_panel, text="模型将实时感知新的几何边界计算共振场", font=('微软雅黑', 8), foreground='gray').pack(anchor=tk.W)
        
        # 画板边框
        canvas_border = ttk.Frame(left_panel, relief="solid", borderwidth=1)
        canvas_border.pack(pady=10)
        
        self.draw_canvas = tk.Canvas(canvas_border, width=256, height=256, bg='black', cursor="pencil")
        self.draw_canvas.pack()
        
        # 内部记录网络输入标准的 64x64 mask
        self.draw_mask = np.zeros((64, 64), dtype=np.float32)
        
        self.draw_canvas.bind('<B1-Motion>', self._paint)
        self.draw_canvas.bind('<Button-1>', self._paint)
        
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="🖌️ 画笔(金属)", command=lambda: self._set_brush_color('white')).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🧽 橡皮(挖洞)", command=lambda: self._set_brush_color('black')).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🗑️ 清空", command=self._clear_canvas).pack(side=tk.RIGHT, padx=2)
        
        self.brush_color = 'white'
        
        ttk.Label(left_panel, text="🎯 测试激发频点 (GHz):", font=('微软雅黑', 10)).pack(anchor=tk.W, pady=(15, 0))
        
        self.draw_freq_var = tk.DoubleVar(value=10.0)
        freq_scale = tk.Scale(left_panel, from_=8.0, to=12.0, resolution=0.1, variable=self.draw_freq_var, orient=tk.HORIZONTAL, showvalue=False, length=200, bg='#f0f0f0', highlightthickness=0)
        freq_scale.pack(fill=tk.X, pady=5)
        
        self.draw_freq_label = ttk.Label(left_panel, text="10.0 GHz", font=('Arial', 10, 'bold'))
        self.draw_freq_label.pack()
        
        def update_label(*args):
            self.draw_freq_label.config(text=f"{self.draw_freq_var.get():.1f} GHz")
        self.draw_freq_var.trace_add("write", update_label)
        
        self.auto_predict_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_panel, text="开启实时自动大模型推演", variable=self.auto_predict_var).pack(anchor=tk.W, pady=10)
        
        big_btn = ttk.Button(left_panel, text="🚀 单次立即预测", command=self._predict_drawn, style='Modern.TButton')
        big_btn.pack(fill=tk.X, pady=10)
        
        # 右侧结果显示区
        right_panel = ttk.Frame(self.draw_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.draw_fig = Figure(figsize=(8, 7), dpi=100)
        self.draw_axs = [self.draw_fig.add_subplot(2, 2, i+1) for i in range(4)]
        for ax in self.draw_axs:
            ax.axis('off')
        self.draw_fig.suptitle("零样本物理场强预测推演区", fontproperties="SimHei", fontsize=16)
        
        self.draw_canvas_plot = FigureCanvasTkAgg(self.draw_fig, right_panel)
        self.draw_canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _set_brush_color(self, color):
        self.brush_color = color

    def _clear_canvas(self):
        self.draw_canvas.delete("all")
        self.draw_mask.fill(0)
        for ax in self.draw_axs:
            ax.clear()
            ax.axis('off')
        self.draw_fig.suptitle("已清空", fontproperties="SimHei", fontsize=16)
        self.draw_canvas_plot.draw()

    def _paint(self, event):
        x, y = event.x, event.y
        r_canvas = 8  # 画笔大小半径
        
        color = self.brush_color
        self.draw_canvas.create_oval(x-r_canvas, y-r_canvas, x+r_canvas, y+r_canvas, fill=color, outline=color)
        
        # 投射到 AI 网格体系
        cx = int(x / 256 * 64)
        cy = int(y / 256 * 64)
        r_mask = max(1, int(r_canvas / 256 * 64))
        
        val = 1.0 if color == 'white' else 0.0
        
        for i in range(max(0, cy-r_mask), min(64, cy+r_mask+1)):
            for j in range(max(0, cx-r_mask), min(64, cx+r_mask+1)):
                if (i-cy)**2 + (j-cx)**2 <= r_mask**2:
                    self.draw_mask[i, j] = val
                    
        # 实时推演极度耗费性能，仅鼠标拖动后如果太频繁可以做防抖，目前简单起见直接调用
        if self.auto_predict_var.get() and np.random.rand() > 0.8: # 降低刷新率保护UI
            self.draw_canvas.after(50, self._predict_drawn)

    def _predict_drawn(self):
        if not self._model:
            messagebox.showwarning("警告", "请先在首页点击【🔄 加载数据和模型】载入您的 AI 权重！")
            self.auto_predict_var.set(False)
            return
            
        mask = self.draw_mask
        if mask.sum() == 0:
            return
            
        try:
            # 计算核心物理映射：从 Mask 计算出流形表面的 SDF
            dist_in = distance_transform_edt(mask)
            dist_out = distance_transform_edt(1 - mask)
            sdf = dist_in - dist_out
            sdf = sdf / (np.abs(sdf).max() + 1e-8)
            
            inp_np = np.stack([sdf, mask], axis=0).astype(np.float32)
            inp = torch.from_numpy(inp_np).unsqueeze(0).to(self._device)
            
            # 【修复点】：模型训练时，频率是被归一化到 [0, 1] 区间的！
            # 计算公示见 dataset.py: (freq - FREQ_MIN) / (FREQ_MAX - FREQ_MIN)
            freq_val = self.draw_freq_var.get()
            freq_norm = (freq_val - 8.0) / (12.0 - 8.0)
            # 手绘区默认用极化 0；数据评估会使用 dataset 中每条的极化条件。
            freq = torch.tensor([freq_norm, 0.0], dtype=torch.float32).to(self._device)
            
            # 因为之前撤销了类别Embedding控制，我们现在直接全靠超网络的 QK 处理形状和频率！
            
            with torch.no_grad():
                pred = self._safe_model_forward(inp, freq)
                pred = self._maybe_denorm_pred(pred)
                p = pred[0].cpu().numpy()
                
                # 分别计算三个方向场强的独立模值
                ex_amp = np.sqrt(p[0]**2 + p[1]**2)
                ey_amp = np.sqrt(p[2]**2 + p[3]**2)
                ez_amp = np.sqrt(p[4]**2 + p[5]**2)
                total_amp = np.sqrt(ex_amp**2 + ey_amp**2 + ez_amp**2 + 1e-8)
                
            for ax in self.draw_axs:
                ax.clear()
                ax.axis('off')
            
            # 使用半透明白线绘制极化形状辅助线
            contour = lambda ax: ax.contour(mask, levels=[0.5], colors='white', alpha=0.7, linewidths=1.2)
            
            # Ex 分量
            self.draw_axs[0].imshow(ex_amp, cmap='jet')
            self.draw_axs[0].set_title(f"|Ex| 模值")
            if mask.sum() > 0: contour(self.draw_axs[0])
            
            # Ey 分量
            self.draw_axs[1].imshow(ey_amp, cmap='jet')
            self.draw_axs[1].set_title(f"|Ey| 模值")
            if mask.sum() > 0: contour(self.draw_axs[1])
            
            # Ez 分量
            self.draw_axs[2].imshow(ez_amp, cmap='jet')
            self.draw_axs[2].set_title(f"|Ez| 模值")
            if mask.sum() > 0: contour(self.draw_axs[2])
            
            # 总场
            self.draw_axs[3].imshow(total_amp, cmap='jet')
            self.draw_axs[3].set_title(f"总场强 |E|")
            if mask.sum() > 0: contour(self.draw_axs[3])
            
            self.draw_fig.suptitle(f"零样本预测物理驻波 @ {freq_val:.2f} GHz", fontproperties="SimHei", fontsize=15)
            self.draw_fig.tight_layout()
            
            self.draw_canvas_plot.draw()
            
        except Exception as e:
            print("预测失败:", e)

    def build_stats_content(self):
        """构建统计信息内容"""
        # 创建滚动文本框显示统计信息
        stats_text_frame = ttk.Frame(self.stats_frame)
        stats_text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.stats_text = tk.Text(stats_text_frame, wrap=tk.WORD, font=('Consolas', 10))
        stats_scrollbar = ttk.Scrollbar(stats_text_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 初始化统计信息内容
        self._init_stats_content()
    
    def _init_stats_content(self):
        """初始化统计信息内容"""
        initial_info = """
📊 模型评估工具
{'='*50}

欢迎使用电磁场预测模型评估工具！

📋 使用步骤:
1. 点击"🔄 加载数据和模型"
2. 等待加载完成
3. 使用各种评估功能

加载完成后，此处将显示详细的配置信息。
"""
        self.stats_text.insert(tk.END, initial_info)
        self.stats_text.configure(state=tk.DISABLED)
    
    def update_stats_info(self):
        """更新统计信息"""
        # 检查stats_text是否存在
        if not hasattr(self, 'stats_text') or self.stats_text is None:
            return

        cfg = lambda k, d='N/A': getattr(C, k, d)

        # 模型信息以实际加载的模型为准（checkpoint 决定架构，config 只是回退）
        model = getattr(self, '_model', None)
        if model is not None:
            arch_name = (getattr(self, '_ckpt_meta', {}) or {}).get("model_name") \
                        or getattr(model, "model_name", "N/A")
            arch_cls = type(model).__name__
            n_params = sum(p.numel() for p in model.parameters())
            model_cfg = C.model_config_from_globals()
            arch_detail = (
                f"    - 注册名: {arch_name} (类 {arch_cls})\n"
                f"    - 参数量: {n_params:,}\n"
                f"    - 模式数 (modes): {model_cfg.modes1} x {model_cfg.modes2}\n"
                f"    - 宽度 (width): {model_cfg.width}\n"
                f"    - 层数 (n_layers): {model_cfg.n_layers}\n"
                f"    - 输出头 (head): {model_cfg.head}\n"
                f"    - 条件化 (kernel_cond): {model_cfg.kernel_cond}"
                + (f" (rank={model_cfg.lora_rank})" if model_cfg.kernel_cond == "lora" else "")
            )
            io_detail = (
                f"  - 输入通道: {model_cfg.in_channels} (SDF + 掩码)\n"
                f"  - 输出通道: {model_cfg.out_channels} "
                f"({'Ex/Ey/Ez/Hx/Hy/Hz 各实虚部' if model_cfg.out_channels >= 12 else '部分分量'})\n"
                f"    - 频率编码维度: {model_cfg.freq_dim}"
            )
        else:
            arch_name, arch_detail = "未加载", "    - 尚未加载模型"
            io_detail = "  - 尚未加载模型"

        self.stats_text.configure(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)

        stats_info = f"""
📊 模型配置信息
{'='*50}

🏗️ 模型架构:
  - 模型类型: {arch_name}
{arch_detail}
{io_detail}

⚙️ 训练配置:
    - 批大小 (batch_size): {cfg('BATCH_SIZE')}
    - 学习率 (learning_rate): {cfg('LR')}
    - 权重衰减 (weight_decay): {cfg('WEIGHT_DECAY')}
    - 验证集比例: {cfg('VAL_RATIO')}
    - 总训练轮次: {cfg('N_EPOCHS')}
    - 早停耐心值: {cfg('EARLY_STOP_PATIENCE')}
    - 预热轮次: {cfg('LR_WARMUP_EPOCHS', 0)}

📁 数据配置:
    - 数据目录: {cfg('DATA_DIR')}
    - 保存目录: {cfg('SAVE_DIR')}
    - 随机种子: {cfg('SHUFFLE_SEED')}

🔧 设备信息:
  - 当前设备: {self._device}
  - CUDA可用: {'是' if torch.cuda.is_available() else '否'}

📈 数据集信息:
"""
        
        if hasattr(self, '_train_ds') and self._train_ds is not None:
            stats_info += f"""  - 训练集样本数: {len(self._train_ds)}
  - 验证集样本数: {len(self._val_ds)}
  - 总样本数: {len(self._train_ds) + len(self._val_ds)}
"""
        else:
            stats_info += "  - 尚未加载数据集\n"
        
        stats_info += f"""
🎯 误差计算说明:
  - DC误差: 直流分量的相对误差
  - AC误差: 交流分量的归一化L2误差  
  - 总误差: (DC误差 + AC误差) / 2
  - 显示格式: 相对误差 (0-1之间的小数)

📊 可用功能:
  ✅ 误差分布分析 - 训练集和验证集的误差直方图
  ✅ 类别-频率热力图 - 不同类别在各频率下的误差
  ✅ 预测结果可视化 - 真实值与预测值的对比
  ✅ 实时状态监控 - 详细的操作日志和进度显示

💡 使用提示:
  1. 先点击"加载数据和模型"
  2. 等待加载完成后使用其他功能
  3. 所有操作都有进度显示和状态反馈
  4. 可以随时清空状态日志
"""
        
        self.stats_text.insert(tk.END, stats_info)
        self.stats_text.configure(state=tk.DISABLED)
    
    def browse_data(self):
        """浏览数据目录"""
        directory = filedialog.askdirectory(initialdir=self.data_var.get())
        if directory:
            self.data_var.set(directory)
            self.refresh_categories()
    
    def browse_checkpoint(self):
        """浏览检查点文件"""
        filename = filedialog.askopenfilename(
            initialdir=os.path.dirname(self.ckpt_var.get()),
            filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            self.ckpt_var.set(filename)
    
    def refresh_categories(self):
        """刷新类别列表"""
        import glob
        root = self.data_var.get()
        if os.path.exists(root):
            cats = sorted([os.path.basename(d) for d in glob.glob(os.path.join(root, "*"))
                          if os.path.isdir(d)])
            self.cat_listbox.delete(0, tk.END)
            for cat in cats:
                self.cat_listbox.insert(tk.END, cat)
            # 默认选择与训练配置一致；若未配置则全选
            self.cat_listbox.select_clear(0, tk.END)
            train_cats = getattr(C, "TRAIN_CATEGORIES", None)
            if train_cats:
                train_set = set(train_cats)
                for idx, cat in enumerate(cats):
                    if cat in train_set:
                        self.cat_listbox.select_set(idx)
                # 若配置类别不在数据目录中，回退为全选，避免空集合
                if not self.cat_listbox.curselection():
                    self.cat_listbox.select_set(0, tk.END)
            else:
                self.cat_listbox.select_set(0, tk.END)
    
    def select_all_cats(self):
        """全选类别"""
        self.cat_listbox.select_set(0, tk.END)
    
    def deselect_all_cats(self):
        """清空类别选择"""
        self.cat_listbox.select_clear(0, tk.END)
    
    def get_selected_categories(self):
        """获取选中的类别"""
        indices = self.cat_listbox.curselection()
        return [self.cat_listbox.get(i) for i in indices]
    
    def clear_status_log(self):
        """清空状态日志"""
        self.status_text.configure(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, "✅ 日志已清空\n")
        self.status_text.configure(state=tk.DISABLED)
    
    def update_status(self, message, is_error=False):
        """更新状态"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # 启用文本编辑
        self.status_text.configure(state=tk.NORMAL)
        
        # 添加时间戳和消息
        if is_error:
            status_line = f"[{timestamp}] ❌ {message}\n"
            self.status_text.insert(tk.END, status_line)
            self.status_text.tag_add("error", f"end-{len(status_line)}c", "end-1c")
            self.status_text.tag_config("error", foreground="#e74c3c", font=('Arial', 9, 'bold'))
        else:
            status_line = f"[{timestamp}] ✅ {message}\n"
            self.status_text.insert(tk.END, status_line)
            self.status_text.tag_add("success", f"end-{len(status_line)}c", "end-1c")
            self.status_text.tag_config("success", foreground="#27ae60", font=('Arial', 9, 'bold'))
        
        # 自动滚动到底部
        self.status_text.see(tk.END)
        
        # 禁用文本编辑
        self.status_text.configure(state=tk.DISABLED)
        
        # 强制更新界面
        self.root.update_idletasks()
        self.root.update()
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_var.set(value)
        self.root.update()

    def _get_selected_dataset(self):
        """根据UI选择返回评估数据集。"""
        mode = self.split_var.get() if hasattr(self, "split_var") else "验证集"
        if self._train_ds is None or self._val_ds is None:
            return None, mode
        if mode == "训练集":
            return self._train_ds, mode
        if mode == "验证集":
            return self._val_ds, mode
        return ConcatDataset([self._train_ds, self._val_ds]), mode

    def _load_checkpoint_robust(self, ckpt_path):
        """按 checkpoint 自身记录的架构重建模型并严格加载权重。

        旧实现是"先把当前 config 构一个模型，再把 checkpoint 里名称与形状都匹配
        的权重塞进去，`strict=False`"，然后只报告"匹配 N 层，跳过 M 层"。这意味着
        模型选错 / 结构参数不对时，GUI 会拿一个**部分随机初始化**的模型画图，
        而界面上看不出任何异常。

        现在：`core.checkpoint.load` 按 checkpoint 里的 `model_config` 重建，
        `strict=True`，不匹配直接抛错。副作用（有意）：默认未指定 checkpoint
        时，GUI 会用当前 config 构建一个随机初始化模型并明确提示，便于先看数据
        和界面；但**一旦指定了 checkpoint，模型类型与结构就完全由它决定**，
        与界面上的 config 无关。
        """
        meta = {"epoch": "N/A", "val_loss": "N/A", "val_rel": "N/A",
                "val_geom": "N/A", "model_name": "N/A", "params": 0,
                "loaded": 0, "skipped": 0}
        self._output_normalizer = None

        if not ckpt_path or not os.path.exists(ckpt_path):
            self.update_status(f"未找到检查点，使用随机初始化模型: {ckpt_path}")
            meta["model_name"] = C.model_name_from_globals()
            meta["params"] = sum(p.numel() for p in self._model.parameters())
            return meta

        try:
            lc = load_checkpoint(ckpt_path, map_location=str(self._device), quiet=True)
        except (KeyError, ValueError, RuntimeError, FileNotFoundError) as e:
            # 不做静默降级：把失败原因完整抛给用户
            self.update_status(f"检查点加载失败: {type(e).__name__}: {e}", True)
            raise

        self._model = lc.model.to(self._device)
        self._output_normalizer = lc.normalizer

        meta["epoch"] = lc.epoch if lc.epoch is not None else "N/A"
        meta["val_loss"] = lc.meta.get("val_loss", "N/A")
        meta["val_rel"] = lc.meta.get("val_rel", "N/A")
        meta["val_geom"] = lc.meta.get("val_geom", "N/A")
        meta["model_name"] = lc.model_name
        meta["params"] = sum(p.numel() for p in self._model.parameters())
        meta["loaded"] = len(lc.model.state_dict())
        meta["skipped"] = 0

        self.update_status(
            f"权重加载完成: {lc.model_name}  参数量 {meta['params']:,}  "
            f"epoch={meta['epoch']}  val_geom={meta['val_geom']}"
        )
        if self._output_normalizer is not None:
            self.update_status(
                f"检测到输出标准化统计量（模式={self._output_normalizer.mode}），"
                "预测结果将自动反标准化"
            )
        return meta

    def _safe_model_forward(self, inp, freq=None, shape_idx=None, type_idx=None):
        """兼容不同 forward 签名。"""
        out = None
        # 模型 forward 只接受 (x, freq)；shape_idx/type_idx 不再参与计算，
        # 保留在签名中仅供旧调用方兼容。
        call_patterns = [
            (inp, freq),
            (inp,),
        ]
        last_err = None
        for args in call_patterns:
            try:
                out = self._model(*args)
                break
            except TypeError as e:
                last_err = e
                continue

        if out is None:
            raise RuntimeError(f"模型前向调用失败: {last_err}")

        if isinstance(out, (tuple, list)):
            out = out[0]
        elif isinstance(out, dict):
            for k in ["pred", "output", "out"]:
                if k in out and torch.is_tensor(out[k]):
                    out = out[k]
                    break
            else:
                tensor_vals = [v for v in out.values() if torch.is_tensor(v)]
                if not tensor_vals:
                    raise RuntimeError("模型输出字典中没有可用 Tensor")
                out = tensor_vals[0]

        if not torch.is_tensor(out):
            raise RuntimeError("模型输出不是 Tensor")
        return out

    def _align_pred_tgt_channels(self, pred, tgt):
        """当模型输出通道数变化时，按最小通道数对齐以保持评估可运行。"""
        if pred.shape[1] == tgt.shape[1]:
            return pred, tgt

        n = min(pred.shape[1], tgt.shape[1])
        if not self._warned_channel_mismatch:
            self.update_status(
                f"检测到通道数不一致: pred={pred.shape[1]}, true={tgt.shape[1]}，已按前 {n} 通道对齐评估",
                True,
            )
            self._warned_channel_mismatch = True
        return pred[:, :n], tgt[:, :n]

    def _maybe_denorm_pred(self, pred):
        if self._output_normalizer is None:
            return pred
        n = min(pred.shape[1], self._output_normalizer.n_channels)
        return self._output_normalizer.denormalize(pred, ch_slice=slice(0, n))
    
    def load_data_model(self):
        """加载数据和模型"""
        if self._running:
            return
        
        def load_worker():
            self._running = True
            try:
                self.update_status("正在加载数据和模型...")
                self.update_progress(10)
                
                selected_cats = self.get_selected_categories()
                if not selected_cats:
                    self.update_status("请至少选择一个类别", True)
                    return
                
                # 加载数据集
                self.update_progress(30)
                freq_s = self.freq_var.get()
                target_freq = None if freq_s == "All" else freq_s.replace(" GHz", "")
                if target_freq:
                    target_freq = f"{float(target_freq):.6f}GHz"
                
                dataset = EMMultiFreqDataset(
                    data_dir=self.data_var.get(),
                    target_freq=target_freq,
                    output_field="all",
                    normalize=False,
                    target_d_over_lambda=getattr(C, "TARGET_D_OVER_LAMBDA", 0.1),
                    use_scatter=getattr(C, "USE_SCATTER_FIELD", False),
                    bg_params_path=os.path.abspath(os.path.join(
                        _HERE, getattr(C, "BACKGROUND_PARAMS_PATH", "../data/blank/background_params.npz")
                    )),
                    polarizations=getattr(C, "DATASET_POLARIZATIONS", None),
                )
                
                # 过滤选中的类别（兼容两级和三级目录结构）
                selected_full_cats = set()
                for cat in selected_cats:
                    for sample in dataset.samples:
                        sample_cat = sample["category"]
                        # 匹配 "cat" 或 "cat/subcat"
                        if sample_cat == cat or sample_cat.startswith(f"{cat}/"):
                            selected_full_cats.add(sample_cat)
                
                dataset.samples = [s for s in dataset.samples
                                 if s["category"] in selected_full_cats]
                
                self.update_progress(50)
                
                # 划分数据集
                self._train_ds, self._val_ds = build_split(dataset)
                
                self.update_progress(70)
                
                # 加载模型（根据 experiment_config 自动选择模型类型）
                self._model = _build_model().to(self._device)

                ckpt_meta = self._load_checkpoint_robust(self.ckpt_var.get())
                self._ckpt_meta = ckpt_meta
                self._model.eval()
                self.update_progress(100)

                quick_rel = float("nan")
                quick_batches = 0
                eval_ds, eval_mode = self._get_selected_dataset()
                # 快速测试：与训练一致的 RelErr 口径（前几批）
                if eval_ds is not None and len(eval_ds) > 0:
                    bs = max(1, min(int(getattr(C, "BATCH_SIZE", 8)), 64))
                    quick_loader = DataLoader(eval_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)
                    max_batches = min(4, len(quick_loader))
                    rel_vals = []

                    with torch.no_grad():
                        for bi, batch in enumerate(quick_loader):
                            inp = batch["input"].to(self._device)
                            tgt = batch["output"].to(self._device)
                            freq = batch["freq"].to(self._device)

                            pred = self._safe_model_forward(inp, freq)
                            pred = self._maybe_denorm_pred(pred)
                            pred, tgt = self._align_pred_tgt_channels(pred, tgt)
                            rel_vals.append(rel_l2_error_batch(pred, tgt))

                            if bi == 0:
                                p0 = pred[0].detach().cpu().numpy()
                                t0 = tgt[0].detach().cpu().numpy()
                                self.update_status(
                                    f"数据范围 - 预测: [{p0.min():.3f}, {p0.max():.3f}], 真实: [{t0.min():.3f}, {t0.max():.3f}]"
                                )

                            quick_batches += 1
                            if quick_batches >= max_batches:
                                break

                    if rel_vals:
                        quick_rel = float(np.mean(rel_vals))
                        if quick_rel > 0.8:
                            self.update_status(f"⚠️ 警告: {eval_mode}快速RelErr较大 ({quick_rel:.4f})", True)
                        elif quick_rel > 0.4:
                            self.update_status(f"⚠️ 注意: {eval_mode}快速RelErr偏高 ({quick_rel:.4f})")
                        else:
                            self.update_status(f"✅ {eval_mode}快速RelErr正常 ({quick_rel:.4f})")

                status_msg = f"""加载成功!
类别: {', '.join(selected_cats)}
训练集: {len(self._train_ds)} 样本
验证集: {len(self._val_ds)} 样本
评估数据集: {eval_mode}
模型: {ckpt_meta['model_name']}  参数量: {ckpt_meta['params']:,}
训练轮次: {ckpt_meta['epoch']}
验证损失(val_loss): {ckpt_meta['val_loss']}
验证误差(val_rel): {ckpt_meta['val_rel']}
验证Geom(val_geom): {ckpt_meta['val_geom']}
快速RelErr(前{quick_batches}批): {quick_rel:.4f}
设备: {self._device}"""

                self.update_status(status_msg)

                # 更新统计信息
                self.update_stats_info()
                    
            except Exception as e:
                self.update_status(f"加载失败: {str(e)}", True)
            finally:
                self._running = False
                self.update_progress(0)
        
        threading.Thread(target=load_worker, daemon=True).start()
    
    def compute_error_distribution(self):
        """计算误差分布"""
        if not self._model:
            messagebox.showwarning("警告", "请先加载数据和模型")
            return
        
        if self._running:
            return
        
        def compute_worker():
            self._running = True
            try:
                self.update_status("正在计算误差分布...")

                mode = self.split_var.get() if hasattr(self, "split_var") else "验证集"
                train_errors = None
                val_errors = None

                if mode in ["训练集", "训练+验证"]:
                    train_errors = self.evaluate_dataset(self._train_ds, "训练集")
                if mode in ["验证集", "训练+验证"]:
                    val_errors = self.evaluate_dataset(self._val_ds, "验证集")
                
                # 绘制误差分布图
                self.plot_error_distribution(train_errors, val_errors)
                
                # 显示与训练一致的总误差
                lines = ["误差分布计算完成", "与训练一致的RelErr:"]
                if train_errors is not None:
                    lines.append(f"  训练集: {train_errors['mean_rel']:.4f}")
                if val_errors is not None:
                    lines.append(f"  验证集: {val_errors['mean_rel']:.4f}")
                lines.append("各分量均值:")
                if train_errors is not None:
                    lines.append(
                        f"  训练集 方向/相位/幅值: {train_errors['dir'].mean():.4f} / {train_errors['phase'].mean():.4f} / {train_errors['mag'].mean():.4f}"
                    )
                if val_errors is not None:
                    lines.append(
                        f"  验证集 方向/相位/幅值: {val_errors['dir'].mean():.4f} / {val_errors['phase'].mean():.4f} / {val_errors['mag'].mean():.4f}"
                    )
                self.update_status("\n".join(lines))
                
            except Exception as e:
                self.update_status(f"计算失败: {str(e)}", True)
            finally:
                self._running = False
                self.update_progress(0)
        
        threading.Thread(target=compute_worker, daemon=True).start()
    
    def evaluate_dataset(self, dataset, label):
        """与训练validate()完全一致的误差计算"""
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=C.BATCH_SIZE, shuffle=False,
                            num_workers=C.NUM_WORKERS, pin_memory=False)

        total_rel_batches = []
        errors = {"dir": [], "phase": [], "mag": []}
        n_batches = len(loader)
        if n_batches == 0:
            self.update_status(f"{label} 为空，跳过评估", True)
            errors["dir"] = np.array([])
            errors["phase"] = np.array([])
            errors["mag"] = np.array([])
            errors["total_rel"] = np.array([])
            errors["mean_rel"] = float("nan")
            return errors

        with torch.no_grad():
            for bi, batch in enumerate(loader):
                self.update_progress((bi + 1) / n_batches * 100)
                self.update_status(f"评估{label}: batch {bi+1}/{n_batches}")

                inp       = batch["input"].to(self._device)
                tgt       = batch["output"].to(self._device)
                freq      = batch["freq"].to(self._device)

                pred = self._safe_model_forward(inp, freq)
                pred = self._maybe_denorm_pred(pred)
                pred, tgt = self._align_pred_tgt_channels(pred, tgt)

                # 与训练完全一致的批量误差
                geom_val, _ = field_geom_loss(pred, tgt)
                total_rel_batches.append(geom_val.item())

                # 逐样本误差（用于分布图）
                for j in range(pred.shape[0]):
                    _, metrics = field_geom_loss(pred[[j]], tgt[[j]])
                    errors["dir"].append(metrics["dir_loss"].item())
                    errors["phase"].append(metrics["phase_loss"].item())
                    errors["mag"].append(metrics["mag_loss"].item())

        for k in errors:
            errors[k] = np.array(errors[k])
        errors["total_rel"] = np.array(total_rel_batches)
        errors["mean_rel"]  = float(np.mean(total_rel_batches))  # 与训练显示完全一致
        return errors
    
    def plot_error_distribution(self, train_errors=None, val_errors=None):
        """绘制误差分布图"""
        for widget in self.error_frame.winfo_children():
            widget.destroy()

        # 4行3列：前两行是Ex/Ey/Ez分量，第3行是与训练一致的总误差
        fig = Figure(figsize=(12, 10), dpi=100)
        axes = fig.subplots(3, 3)
        fig.suptitle('误差分布对比 (Geom)', fontsize=14, fontweight='bold')

        components = ["dir", "phase", "mag"]
        comp_names = {"dir": "方向角", "phase": "相位", "mag": "幅值"}
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        labels = ["训练集", "验证集"]

        for i, comp in enumerate(components):
            for row, (errors, lbl) in enumerate([(train_errors, "训练集"), (val_errors, "验证集")]):
                ax = axes[row, i]
                if errors is None or comp not in errors or len(errors[comp]) == 0:
                    ax.set_title(f'{lbl} {comp_names[comp]} 误差')
                    ax.text(0.5, 0.5, '未选择/无数据', ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                e = errors[comp]
                ax.hist(e, bins=40, alpha=0.7, color=colors[i], edgecolor='black', linewidth=0.5)
                ax.axvline(e.mean(), color='red', linestyle='--', label=f'均值: {e.mean():.3f}')
                ax.axvline(np.median(e), color='orange', linestyle=':', label=f'中位: {np.median(e):.3f}')
                ax.set_title(f'{lbl} {comp_names[comp]} 误差')
                ax.set_xlabel('相对误差')
                ax.set_ylabel('频次')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        # 第3行：与训练一致的总误差对比
        for i, (errors, lbl, color) in enumerate([(train_errors, "训练集", '#3498db'),
                                                   (val_errors,   "验证集", '#e74c3c')]):
            ax = axes[2, i] if i < 2 else axes[2, 2]
            if errors is None or "total_rel" not in errors or len(errors["total_rel"]) == 0:
                ax.set_title(f'{lbl} 总误差 (与训练一致)')
                ax.text(0.5, 0.5, '未选择/无数据', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            e = errors["total_rel"]
            ax.hist(e, bins=40, alpha=0.7, color=color, edgecolor='black', linewidth=0.5)
            ax.axvline(e.mean(), color='red', linestyle='--', label=f'均值: {e.mean():.3f}')
            ax.axvline(np.median(e), color='orange', linestyle=':', label=f'中位: {np.median(e):.3f}')
            ax.set_title(f'{lbl} 总误差 (与训练一致)')
            ax.set_xlabel('相对误差')
            ax.set_ylabel('频次')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # 第3行第3列：训练集vs验证集总误差对比
        ax = axes[2, 2]
        has_train = train_errors is not None and "total_rel" in train_errors and len(train_errors["total_rel"]) > 0
        has_val = val_errors is not None and "total_rel" in val_errors and len(val_errors["total_rel"]) > 0
        if has_train:
            ax.hist(train_errors["total_rel"], bins=40, alpha=0.6, color='#3498db', label=f'训练 {train_errors["total_rel"].mean():.3f}')
        if has_val:
            ax.hist(val_errors["total_rel"], bins=40, alpha=0.6, color='#e74c3c', label=f'验证 {val_errors["total_rel"].mean():.3f}')
        if has_train or has_val:
            ax.set_title('总误差对比')
            ax.set_xlabel('相对误差')
            ax.set_ylabel('频次')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.set_title('总误差对比')
            ax.text(0.5, 0.5, '未选择/无数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.error_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def generate_heatmap(self):
        """生成热力图"""
        if not self._model:
            messagebox.showwarning("警告", "请先加载数据和模型")
            return

        dataset, mode = self._get_selected_dataset()
        if dataset is None or len(dataset) == 0:
            messagebox.showwarning("警告", f"{mode}为空，无法生成热力图")
            return
        
        if self._running:
            return
        
        def heatmap_worker():
            self._running = True
            try:
                self.update_status("正在生成类别-频率热力图...")
                
                # 使用选择的数据集生成热力图
                dataset, mode = self._get_selected_dataset()
                
                # 按类别和频率分组计算误差
                category_freq_total = defaultdict(lambda: defaultdict(list))
                category_freq_dir = defaultdict(lambda: defaultdict(list))
                category_freq_phase = defaultdict(lambda: defaultdict(list))
                category_freq_mag = defaultdict(lambda: defaultdict(list))
                
                with torch.no_grad():
                    for i in range(len(dataset)):
                        progress = (i + 1) / len(dataset) * 100
                        self.update_progress(progress)
                        self.update_status(f"计算热力图数据: {i+1}/{len(dataset)}")
                        
                        s = dataset[i]
                        category = s["category"]
                        freq_ghz = s["freq_ghz"]
                        
                        inp = s["input"].unsqueeze(0).to(self._device)
                        tgt = s["output"].unsqueeze(0).to(self._device)
                        freq = s["freq"].unsqueeze(0).to(self._device)

                        pred = self._safe_model_forward(inp, freq)
                        pred = self._maybe_denorm_pred(pred)
                        pred, tgt = self._align_pred_tgt_channels(pred, tgt)
                        p = pred[0].cpu().numpy()
                        t = tgt[0].cpu().numpy()
                        
                        # 使用 geom 指标（方向角+相位+幅值）与训练一致
                        geom_val, geom_components = field_geom_loss(pred, tgt)
                        
                        category_freq_total[category][freq_ghz].append(geom_val.item())
                        category_freq_dir[category][freq_ghz].append(geom_components["dir_loss"].item())
                        category_freq_phase[category][freq_ghz].append(geom_components["phase_loss"].item())
                        category_freq_mag[category][freq_ghz].append(geom_components["mag_loss"].item())
                
                # 绘制热力图
                self.plot_category_frequency_heatmap({
                    "Total Geom Loss": category_freq_total,
                    "Direction Angle Loss": category_freq_dir,
                    "Complex Phase Loss": category_freq_phase,
                    "Relative Mag Loss": category_freq_mag
                })
                
                self.update_status(f"✅ 热力图生成完成（数据集: {mode}）")
                
            except Exception as e:
                self.update_status(f"热力图生成失败: {str(e)}", True)
            finally:
                self._running = False
                self.update_progress(0)
        
        threading.Thread(target=heatmap_worker, daemon=True).start()
    
    def plot_category_frequency_heatmap(self, multi_heatmap_dicts):
        """绘制单一热力图，并在格子里显示方向、相位、幅值的细分误差"""
        # 清除之前的图
        for widget in self.heatmap_frame.winfo_children():
            widget.destroy()
        
        # 取 Total 字典 获取 dimensions
        sample_dict = multi_heatmap_dicts["Total Geom Loss"]
        categories = sorted(sample_dict.keys())
        frequencies = sorted(set(freq for cat_errors in sample_dict.values() 
                               for freq in cat_errors.keys()))
        
        heatmap_data = []
        for cat in categories:
            row = []
            for freq in frequencies:
                if freq in sample_dict[cat]:
                    # 热力图的底色由 Total Geom Loss 决定
                    row.append(np.mean(sample_dict[cat][freq]))
                else:
                    row.append(np.nan)
            heatmap_data.append(row)
            
        fig = Figure(figsize=(12, 8), dpi=100) # 稍微加长宽一点以容纳多行文字
        ax = fig.add_subplot(111)
        
        im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
        
        ax.set_xticks(range(len(frequencies)))
        ax.set_xticklabels([f"{f:.1f}" for f in frequencies])
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)
        
        ax.set_xlabel('频率 (GHz)')
        ax.set_ylabel('类别')
        ax.set_title('误差剖析热力图 (底色:Geom总误差 | D:方向角误差 | P:相位误差 | M:幅值误差)')
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Total Geom Loss (值越小越好)')
        
        # 在每个格子中显示多行数值
        for i in range(len(categories)):
            cat = categories[i]
            for j in range(len(frequencies)):
                freq = frequencies[j]
                if not np.isnan(heatmap_data[i][j]):
                    dir_err = np.mean(multi_heatmap_dicts["Direction Angle Loss"][cat][freq])
                    pha_err = np.mean(multi_heatmap_dicts["Complex Phase Loss"][cat][freq])
                    mag_err = np.mean(multi_heatmap_dicts["Relative Mag Loss"][cat][freq])
                    
                    val = heatmap_data[i][j]
                    text_str = f"D: {dir_err:.3f}\nP: {pha_err:.3f}\nM: {mag_err:.3f}"
                    
                    # 最痛点的极端误差颜色偏黄，字体改成黑的看得清楚
                    font_color = "black" if val > np.nanmax(heatmap_data) * 0.7 else "white"
                    ax.text(j, i, text_str, ha="center", va="center", color=font_color, fontsize=8)
        
        fig.tight_layout()
        
        # 嵌入到tkinter
        canvas = FigureCanvasTkAgg(fig, self.heatmap_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def generate_field_stats_heatmap(self):
        """生成全集数据的物理场强均值热力图（计算真实的|E|，无需模型推理）"""
        if self._train_ds is None or self._val_ds is None:
            messagebox.showwarning("警告", "请先加载数据")
            return
            
        if self._running:
            return
            
        def stats_worker():
            self._running = True
            try:
                self.update_status("正在统计全局数据真实的场强均值热力图...")
                
                # 无论选择什么，都用全体数据
                dataset = ConcatDataset([self._train_ds, self._val_ds])
                
                # 记录所有样本的场强数据
                # 列表元素为: (category, freq_ghz, mean_amp)
                all_samples_stats = []
                
                with torch.no_grad():
                    for i in range(len(dataset)):
                        if i % 10 == 0:
                            progress = (i + 1) / len(dataset) * 100
                            self.update_progress(progress)
                            self.update_status(f"统计场强数据: {i+1}/{len(dataset)}")
                        
                        s = dataset[i]
                        # 真值 (6, H, W)
                        tgt = s["output"]
                        
                        ex = tgt[0]**2 + tgt[1]**2
                        ey = tgt[2]**2 + tgt[3]**2
                        ez = tgt[4]**2 + tgt[5]**2
                        total_amp = torch.sqrt(ex + ey + ez + 1e-6)
                        
                        # 我们使用场总强度的空间最大值作为离群指标（能完美抓住尖峰奇异点）
                        # 也可使用 mean，但 max 更能体现极化突变的离群
                        max_amp = total_amp.max().item()
                        
                        all_samples_stats.append({
                            "cat": s["category"],
                            "freq": s["freq_ghz"],
                            "val": max_amp
                        })
                
                # 开始统计离群值（前 10% 和 后 10%）
                self.update_status("正在计算全集前 10% 与后 10% 离群占比...")
                vals = [s["val"] for s in all_samples_stats]
                vals.sort()
                n_total = len(vals)
                p10_idx = int(n_total * 0.1)
                p90_idx = int(n_total * 0.9)
                
                val_low_th = vals[p10_idx]
                val_high_th = vals[p90_idx]
                
                # 统计每个 (cat, freq) 极光格子里的总数 和 离群数
                grid_stats = defaultdict(lambda: {"total": 0, "low_outliers": 0, "high_outliers": 0})
                
                for s in all_samples_stats:
                    k = (s["cat"], s["freq"])
                    grid_stats[k]["total"] += 1
                    if s["val"] <= val_low_th:
                        grid_stats[k]["low_outliers"] += 1
                    if s["val"] >= val_high_th:
                        grid_stats[k]["high_outliers"] += 1
                
                # 绘制分布热力图
                self.plot_field_stats_heatmap(grid_stats)
                
                # 自动跳过去
                self.notebook.select(self.field_stats_frame)
                self.update_status(f"✅ 离群值热力图生成完成（全集样本数: {n_total}）")
                
            except Exception as e:
                self.update_status(f"统计失败: {str(e)}", True)
            finally:
                self._running = False
                self.update_progress(0)
        
        threading.Thread(target=stats_worker, daemon=True).start()

    def plot_field_stats_heatmap(self, grid_stats):
        """绘制最低 10% 和最高 10% 离群样本占比的三维热力图"""
        for widget in self.field_stats_frame.winfo_children():
            widget.destroy()
            
        categories = sorted(list(set([k[0] for k in grid_stats.keys()])))
        frequencies = sorted(list(set([k[1] for k in grid_stats.keys()])))
                               
        low_data = []
        high_data = []
        
        for cat in categories:
            row_low = []
            row_high = []
            for freq in frequencies:
                k = (cat, freq)
                if grid_stats[k]["total"] > 0:
                    pct_low = (grid_stats[k]["low_outliers"] / grid_stats[k]["total"]) * 100
                    pct_high = (grid_stats[k]["high_outliers"] / grid_stats[k]["total"]) * 100
                    row_low.append(pct_low)
                    row_high.append(pct_high)
                else:
                    row_low.append(np.nan)
                    row_high.append(np.nan)
            low_data.append(row_low)
            high_data.append(row_high)
            
        fig = Figure(figsize=(14, 6), dpi=100)
        
        # 左图：最低 10% 占比
        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(low_data, cmap='Blues', aspect='auto', vmin=0, vmax=100)
        ax1.set_xticks(range(len(frequencies)))
        ax1.set_xticklabels([f"{f:.1f}" for f in frequencies])
        ax1.set_yticks(range(len(categories)))
        ax1.set_yticklabels(categories)
        ax1.set_title('全集最低 10% 弱场强样本的类别集中度 (%)')
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        for i in range(len(categories)):
            for j in range(len(frequencies)):
                if not np.isnan(low_data[i][j]) and low_data[i][j] > 0:
                    ax1.text(j, i, f'{low_data[i][j]:.1f}%', ha="center", va="center", color="black", fontsize=8)

        # 右图：最高 10% 占比
        ax2 = fig.add_subplot(122)
        im2 = ax2.imshow(high_data, cmap='Reds', aspect='auto', vmin=0, vmax=100)
        ax2.set_xticks(range(len(frequencies)))
        ax2.set_xticklabels([f"{f:.1f}" for f in frequencies])
        ax2.set_yticks(range(len(categories)))
        ax2.set_yticklabels(categories)
        ax2.set_title('全集最高 10% 极端强场样本的类别集中度 (%)')
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        for i in range(len(categories)):
            for j in range(len(frequencies)):
                if not np.isnan(high_data[i][j]) and high_data[i][j] > 0:
                    ax2.text(j, i, f'{high_data[i][j]:.1f}%', ha="center", va="center", color="black" if high_data[i][j]<50 else "white", fontsize=8)
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, self.field_stats_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def visualize_predictions(self):
        """可视化预测结果"""
        if not self._model:
            messagebox.showwarning("警告", "请先加载数据和模型")
            return

        dataset, mode = self._get_selected_dataset()
        if dataset is None or len(dataset) == 0:
            messagebox.showwarning("警告", f"{mode}为空，无法可视化")
            return
        
        if self._running:
            return
        
        def visualize_worker():
            self._running = True
            try:
                self.update_status("正在生成预测可视化（幅值页 + 复数页）...")
                
                # 使用选择数据集的样本
                dataset, mode = self._get_selected_dataset()
                n_samples = min(4, len(dataset))
                indices = np.random.choice(len(dataset), size=n_samples, replace=False)

                items = []
                
                with torch.no_grad():
                    for row, idx in enumerate(indices):
                        progress = (row + 1) / n_samples * 100
                        self.update_progress(progress)
                        self.update_status(f"处理样本: {row+1}/{n_samples}")
                        
                        s = dataset[int(idx)]
                        inp = s["input"].unsqueeze(0).to(self._device)
                        tgt = s["output"].unsqueeze(0).to(self._device)
                        freq = s["freq"].unsqueeze(0).to(self._device)

                        pred = self._safe_model_forward(inp, freq)
                        pred = self._maybe_denorm_pred(pred)
                        pred, tgt = self._align_pred_tgt_channels(pred, tgt)
                        p = pred[0].cpu().numpy()
                        t = tgt[0].cpu().numpy()

                        items.append({
                            "sample": s,
                            "mask": inp[0, 1].cpu().numpy(),
                            "pred": p,
                            "true": t,
                        })

                self._plot_amplitude_error_page(items, mode)
                self._plot_complex_error_page(items, mode)
                self.notebook.select(self.viz_amp_frame)
                self.update_status(f"✅ 预测可视化完成（数据集: {mode}，已生成两页）")
                
            except Exception as e:
                self.update_status(f"可视化失败: {str(e)}", True)
            finally:
                self._running = False
                self.update_progress(0)
        
        threading.Thread(target=visualize_worker, daemon=True).start()

    def _plot_amplitude_error_page(self, items, mode):
        """页1：显示幅值(True/Pred/|Err|)，电场+磁场分开显示。"""
        for widget in self.viz_amp_frame.winfo_children():
            widget.destroy()

        n_samples = len(items)
        # 根据实际通道数决定显示电场还是电磁场
        n_ch = items[0]["pred"].shape[0] if items else 6
        has_mag = n_ch >= 12

        comp_names    = ["Ex", "Ey", "Ez"]
        comp_channels = [0, 2, 4]
        if has_mag:
            comp_names    += ["Hx", "Hy", "Hz"]
            comp_channels += [6, 8, 10]

        n_comps = len(comp_names)
        n_cols = 1 + n_comps * 3  # mask列 + 每分量3列(True/Pred/Err)

        fig = Figure(figsize=(max(20, n_cols * 2.0), max(3.2 * n_samples, 4.0)), dpi=100)
        fig.suptitle(f"幅值误差可视化 ({mode})", fontsize=12, fontweight='bold')
        axes = fig.subplots(n_samples, n_cols, squeeze=False)
        fig.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.03, wspace=0.06, hspace=0.35)

        for r, item in enumerate(items):
            s = item["sample"]
            p = item["pred"]
            t = item["true"]

            axes[r, 0].imshow(item["mask"], cmap="gray", origin="lower")
            axes[r, 0].set_title(f"{s['name']}\n{s['category']}", fontsize=8)
            axes[r, 0].axis("off")

            for i, (name, ch) in enumerate(zip(comp_names, comp_channels)):
                c0 = 1 + i * 3
                t_amp = np.sqrt(t[ch] ** 2 + t[ch + 1] ** 2)
                p_amp = np.sqrt(p[ch] ** 2 + p[ch + 1] ** 2)
                e_amp = np.abs(p_amp - t_amp)

                vmin = float(min(t_amp.min(), p_amp.min()))
                vmax = float(max(t_amp.max(), p_amp.max()))
                if vmax <= vmin:
                    vmax = vmin + 1e-6

                amp_err = amp_l2_error(p, t, ch)

                axes[r, c0].imshow(t_amp, cmap="jet", vmin=vmin, vmax=vmax, origin="lower")
                axes[r, c0].set_title(f"|{name}| True", fontsize=8)
                axes[r, c0].axis("off")

                axes[r, c0 + 1].imshow(p_amp, cmap="jet", vmin=vmin, vmax=vmax, origin="lower")
                axes[r, c0 + 1].set_title(f"|{name}| Pred\n{amp_err:.1%}", fontsize=8)
                axes[r, c0 + 1].axis("off")

                axes[r, c0 + 2].imshow(e_amp, cmap="hot", origin="lower")
                axes[r, c0 + 2].set_title(f"|{name}| |Err|", fontsize=8)
                axes[r, c0 + 2].axis("off")

        canvas = FigureCanvasTkAgg(fig, self.viz_amp_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_complex_error_page(self, items, mode):
        """页2：显示实部/虚部(True/Pred/|Err|)误差，电场+磁场分开显示。"""
        for widget in self.viz_complex_frame.winfo_children():
            widget.destroy()

        n_samples = len(items)
        n_ch = items[0]["pred"].shape[0] if items else 6
        has_mag = n_ch >= 12

        dir_names         = ["Ex", "Ey", "Ez"]
        dir_base_channels = [0, 2, 4]
        if has_mag:
            dir_names         += ["Hx", "Hy", "Hz"]
            dir_base_channels += [6, 8, 10]

        n_dirs = len(dir_names)
        n_cols = 1 + n_dirs * 3

        fig = Figure(figsize=(max(20, n_cols * 2.0), max(5.8 * n_samples, 6.0)), dpi=100)
        fig.suptitle(f"复数实部/虚部误差可视化 ({mode})", fontsize=12, fontweight='bold')
        axes = fig.subplots(n_samples * 2, n_cols, squeeze=False)
        fig.subplots_adjust(left=0.02, right=0.99, top=0.94, bottom=0.02, wspace=0.06, hspace=0.35)

        for s_idx, item in enumerate(items):
            s = item["sample"]
            p = item["pred"]
            t = item["true"]

            row_r = s_idx * 2
            row_i = row_r + 1

            axes[row_r, 0].imshow(item["mask"], cmap="gray", origin="lower")
            axes[row_r, 0].set_title(f"{s['name']}\n{s['category']}", fontsize=8)
            axes[row_r, 0].axis("off")

            axes[row_i, 0].text(0.5, 0.5, "Imag", ha="center", va="center", fontsize=9)
            axes[row_i, 0].axis("off")

            for d_idx, (d_name, base_ch) in enumerate(zip(dir_names, dir_base_channels)):
                col0 = 1 + d_idx * 3
                for part_name, row, ch in [("Re", row_r, base_ch), ("Im", row_i, base_ch + 1)]:
                    t_map = t[ch]
                    p_map = p[ch]
                    e_map = np.abs(p_map - t_map)

                    vmin = float(min(t_map.min(), p_map.min()))
                    vmax = float(max(t_map.max(), p_map.max()))
                    if vmax <= vmin:
                        vmax = vmin + 1e-6

                    rel = np.linalg.norm(p_map - t_map) / (np.linalg.norm(t_map) + 1e-6)

                    axes[row, col0].imshow(t_map, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
                    axes[row, col0].set_title(f"{d_name}_{part_name} True", fontsize=7)
                    axes[row, col0].axis("off")

                    axes[row, col0 + 1].imshow(p_map, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
                    axes[row, col0 + 1].set_title(f"{d_name}_{part_name} Pred\n{rel:.1%}", fontsize=7)
                    axes[row, col0 + 1].axis("off")

                    axes[row, col0 + 2].imshow(e_map, cmap="hot", origin="lower")
                    axes[row, col0 + 2].set_title(f"{d_name}_{part_name} |Err|", fontsize=7)
                    axes[row, col0 + 2].axis("off")

        canvas = FigureCanvasTkAgg(fig, self.viz_complex_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def diagnose_data(self):
        """数据诊断功能"""
        if not self._model:
            messagebox.showwarning("警告", "请先加载数据和模型")
            return

        dataset, mode = self._get_selected_dataset()
        if dataset is None or len(dataset) == 0:
            messagebox.showwarning("警告", f"{mode}为空，无法诊断")
            return
        
        if self._running:
            return
        
        def diagnose_worker():
            self._running = True
            try:
                self.update_status("正在进行数据诊断...")
                
                # 检查几个样本的数据范围和误差
                dataset, mode = self._get_selected_dataset()
                n_check = min(10, len(dataset))
                
                errors_list = []
                pred_ranges = []
                true_ranges = []
                
                with torch.no_grad():
                    for i in range(n_check):
                        progress = (i + 1) / n_check * 100
                        self.update_progress(progress)
                        
                        s = dataset[i]
                        inp = s["input"].unsqueeze(0).to(self._device)
                        tgt = s["output"].unsqueeze(0).to(self._device)
                        freq = s["freq"].unsqueeze(0).to(self._device)

                        pred = self._safe_model_forward(inp, freq)
                        pred = self._maybe_denorm_pred(pred)
                        pred, tgt = self._align_pred_tgt_channels(pred, tgt)
                        p = pred[0].cpu().numpy()
                        t = tgt[0].cpu().numpy()
                        
                        # 记录数据范围
                        pred_ranges.append((p.min(), p.max()))
                        true_ranges.append((t.min(), t.max()))
                        
                        # 计算误差
                        err = (dc_ac_error(p, t, 0) + dc_ac_error(p, t, 2) + dc_ac_error(p, t, 4)) / 3
                        errors_list.append(err)
                
                # 分析结果
                avg_error = np.mean(errors_list)
                pred_min = min(r[0] for r in pred_ranges)
                pred_max = max(r[1] for r in pred_ranges)
                true_min = min(r[0] for r in true_ranges)
                true_max = max(r[1] for r in true_ranges)
                
                diagnosis = f"""
📋 数据诊断报告
{'='*30}

📊 误差统计 (检查{n_check}个样本):
  平均误差: {avg_error:.4f}
  最小误差: {min(errors_list):.4f}
  最大误差: {max(errors_list):.4f}

📈 数据范围:
  预测值范围: [{pred_min:.3f}, {pred_max:.3f}]
  真实值范围: [{true_min:.3f}, {true_max:.3f}]

🔍 诊断结果:"""
                
                if avg_error > 2.0:
                    diagnosis += "\n  ❌ 误差过大，可能原因："
                    diagnosis += "\n     - 模型未充分训练"
                    diagnosis += "\n     - 数据预处理问题"
                    diagnosis += "\n     - 模型架构不匹配"
                elif avg_error > 1.0:
                    diagnosis += "\n  ⚠️ 误差偏高，建议："
                    diagnosis += "\n     - 检查训练过程"
                    diagnosis += "\n     - 调整超参数"
                elif avg_error > 0.2:
                    diagnosis += "\n  ✅ 误差在可接受范围内"
                else:
                    diagnosis += "\n  🎯 误差很小，模型表现良好"
                
                if abs(pred_max - true_max) > abs(true_max) * 0.5:
                    diagnosis += "\n  ⚠️ 预测值和真实值范围差异较大"
                
                self.update_status(diagnosis + f"\n\n📌 当前数据集: {mode}")
                
            except Exception as e:
                self.update_status(f"诊断失败: {str(e)}", True)
            finally:
                self._running = False
                self.update_progress(0)
        
        threading.Thread(target=diagnose_worker, daemon=True).start()
    
    def run(self):
        """运行应用"""
        self.root.mainloop()

def main():
    app = EvalApp()
    app.run()

if __name__ == "__main__":
    main()
