#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
经典CNN模型对比GUI程序
作者：范宇桐、栾华糠
调用现有架构：
- models/ 目录下的预定义CNN模型
- experiments/proper_grid_search.py 中的数据加载和训练逻辑
"""

import os
import sys
import json
import threading
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

warnings.filterwarnings('ignore')

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'experiments'))

# 导入现有模型
from models.LeNet_5 import LeNet5
from models.AlexNet import AlexNet
from models.VGG import VGG11, VGG13, VGG16, VGG19
from models.ResNet import ResNet18, ResNet34, ResNet50
from models.Inception import InceptionV1, InceptionV3
from models.DenseNet import DenseNet121, DenseNet169

# 模型类映射
MODEL_MAP = {
    'LeNet5': LeNet5,
    'AlexNet': AlexNet,
    'VGG11': VGG11,
    'VGG13': VGG13,
    'VGG16': VGG16,
    'VGG19': VGG19,
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'InceptionV1': InceptionV1,
    'InceptionV3': InceptionV3,
    'DenseNet121': DenseNet121,
    'DenseNet169': DenseNet169
}

# 默认参数空间
DEFAULT_PARAM_SPACE = {
    'LeNet5': {'batch_size': [32, 64], 'learning_rate': [0.001, 0.005], 'weight_decay': [0.0, 0.0001]},
    'AlexNet': {'batch_size': [16, 32, 64], 'learning_rate': [0.0001, 0.001], 'weight_decay': [0.0, 0.0001, 0.001]},
    'VGG11': {'batch_size': [16, 32], 'learning_rate': [0.0005, 0.001], 'weight_decay': [0.00005, 0.0001]},
    'VGG13': {'batch_size': [16, 32], 'learning_rate': [0.0005, 0.001], 'weight_decay': [0.00005, 0.0001]},
    'VGG16': {'batch_size': [16, 32], 'learning_rate': [0.0005, 0.001, 0.002], 'weight_decay': [0.00005, 0.0001]},
    'VGG19': {'batch_size': [16, 32], 'learning_rate': [0.0005, 0.001], 'weight_decay': [0.00005, 0.0001]},
    'ResNet18': {'batch_size': [16, 32, 64], 'learning_rate': [0.0001, 0.0005, 0.001], 'weight_decay': [0.0, 0.00005, 0.0001]},
    'ResNet34': {'batch_size': [16, 32], 'learning_rate': [0.0001, 0.0005], 'weight_decay': [0.0, 0.0001]},
    'ResNet50': {'batch_size': [16, 32], 'learning_rate': [0.0001, 0.0005], 'weight_decay': [0.0, 0.0001]},
    'InceptionV1': {'batch_size': [32, 64], 'learning_rate': [0.001, 0.005], 'weight_decay': [0.0, 0.0001]},
    'InceptionV3': {'batch_size': [16, 32], 'learning_rate': [0.001, 0.005], 'weight_decay': [0.0, 0.0001]},
    'DenseNet121': {'batch_size': [16, 32], 'learning_rate': [0.001, 0.005], 'weight_decay': [0.0, 0.0001]},
    'DenseNet169': {'batch_size': [16, 32], 'learning_rate': [0.001, 0.005], 'weight_decay': [0.0, 0.0001]}
}


class CNNComparisonGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("经典CNN模型对比 - 色盲测试图表分类 - 范宇桐、栾华糠")
        self.root.geometry("1400x900")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_names = None
        self.training_history = None
        self.is_training = False
        
        self.setup_ui()
        self.log(f"程序启动，运行设备: {self.device}")
    
    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # 标签页
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 1. 数据管理标签页
        data_tab = ttk.Frame(notebook, padding="10")
        notebook.add(data_tab, text="数据管理")
        self.setup_data_tab(data_tab)
        
        # 2. 模型训练标签页
        train_tab = ttk.Frame(notebook, padding="10")
        notebook.add(train_tab, text="模型训练")
        self.setup_train_tab(train_tab)
        
        # 3. 结果查看标签页
        results_tab = ttk.Frame(notebook, padding="10")
        notebook.add(results_tab, text="实验结果")
        self.setup_results_tab(results_tab)
        
        # 4. 预测识别标签页
        predict_tab = ttk.Frame(notebook, padding="10")
        notebook.add(predict_tab, text="预测识别")
        self.setup_predict_tab(predict_tab)
        
        # 日志区域
        log_frame = ttk.LabelFrame(main_frame, text="日志输出", padding="5")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        main_frame.rowconfigure(1, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_data_tab(self, parent):
        frame = ttk.LabelFrame(parent, text="数据集管理", padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 数据集路径
        path_frame = ttk.Frame(frame)
        path_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(path_frame, text="数据集路径:").pack(side=tk.LEFT)
        self.data_path_var = tk.StringVar(value=str(PROJECT_ROOT / 'data' / 'ishihara_dataset' / 'train'))
        ttk.Entry(path_frame, textvariable=self.data_path_var, width=60).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(path_frame, text="浏览...", command=self.browse_data_path).pack(side=tk.LEFT, padx=5)
        ttk.Button(path_frame, text="加载数据集", command=self.load_dataset).pack(side=tk.LEFT, padx=5)
        
        # 数据集信息
        info_frame = ttk.LabelFrame(frame, text="数据集信息", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.data_info_text = scrolledtext.ScrolledText(info_frame, height=15, wrap=tk.WORD)
        self.data_info_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_train_tab(self, parent):
        # 左侧：模型选择和参数配置
        left_frame = ttk.Frame(parent)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5)
        
        # 模型选择
        model_frame = ttk.LabelFrame(left_frame, text="模型选择", padding="10")
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="选择模型:").pack(anchor=tk.W)
        self.model_var = tk.StringVar(value='LeNet5')
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=list(MODEL_MAP.keys()), state='readonly')
        model_combo.pack(fill=tk.X, pady=5)
        model_combo.bind('<<ComboboxSelected>>', self.on_model_selected)
        
        # 参数配置
        param_frame = ttk.LabelFrame(left_frame, text="训练参数", padding="10")
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="批次大小:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Entry(param_frame, textvariable=self.batch_size_var, width=15).grid(row=0, column=1, pady=2)
        
        ttk.Label(param_frame, text="学习率:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Entry(param_frame, textvariable=self.lr_var, width=15).grid(row=1, column=1, pady=2)
        
        ttk.Label(param_frame, text="权重衰减:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.weight_decay_var = tk.DoubleVar(value=0.0001)
        ttk.Entry(param_frame, textvariable=self.weight_decay_var, width=15).grid(row=2, column=1, pady=2)
        
        ttk.Label(param_frame, text="训练轮数:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.IntVar(value=50)
        ttk.Entry(param_frame, textvariable=self.epochs_var, width=15).grid(row=3, column=1, pady=2)
        
        # 训练控制
        control_frame = ttk.LabelFrame(left_frame, text="训练控制", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        self.train_button = ttk.Button(control_frame, text="开始训练", command=self.start_training_thread)
        self.train_button.pack(fill=tk.X, pady=5)
        
        self.stop_button = ttk.Button(control_frame, text="停止训练", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=5)
        
        # 进度条
        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100).pack(fill=tk.X, pady=5)
        self.status_label = ttk.Label(control_frame, text="就绪")
        self.status_label.pack(pady=5)
        
        # 右侧：训练结果和图表
        right_frame = ttk.Frame(parent)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # 训练历史
        history_frame = ttk.LabelFrame(right_frame, text="训练历史", padding="10")
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        self.history_text = scrolledtext.ScrolledText(history_frame, height=10, wrap=tk.WORD)
        self.history_text.pack(fill=tk.BOTH, expand=True)
        
        # 图表显示
        plot_frame = ttk.LabelFrame(right_frame, text="训练曲线", padding="10")
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.plot_label = ttk.Label(plot_frame, text="训练完成后将显示图表")
        self.plot_label.pack(fill=tk.BOTH, expand=True)
        
        # 保存模型按钮
        ttk.Button(right_frame, text="保存模型", command=self.save_model).pack(pady=5)
    
    def setup_results_tab(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 结果目录选择
        dir_frame = ttk.Frame(frame)
        dir_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(dir_frame, text="结果目录:").pack(side=tk.LEFT)
        self.results_path_var = tk.StringVar(value=str(PROJECT_ROOT / 'proper_grid_results'))
        ttk.Entry(dir_frame, textvariable=self.results_path_var, width=60).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(dir_frame, text="浏览...", command=self.browse_results_path).pack(side=tk.LEFT, padx=5)
        ttk.Button(dir_frame, text="刷新列表", command=self.refresh_results_list).pack(side=tk.LEFT, padx=5)
        
        # 模型列表
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(list_frame, text="已训练模型:").pack(anchor=tk.W)
        
        self.results_tree = ttk.Treeview(list_frame, columns=('model', 'mean_acc', 'std_acc', 'max_acc', 'min_acc'), show='headings')
        self.results_tree.heading('model', text='模型')
        self.results_tree.heading('mean_acc', text='平均准确率')
        self.results_tree.heading('std_acc', text='标准差')
        self.results_tree.heading('max_acc', text='最高准确率')
        self.results_tree.heading('min_acc', text='最低准确率')
        
        self.results_tree.column('model', width=120)
        self.results_tree.column('mean_acc', width=100)
        self.results_tree.column('std_acc', width=100)
        self.results_tree.column('max_acc', width=100)
        self.results_tree.column('min_acc', width=100)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 详情显示
        detail_frame = ttk.LabelFrame(frame, text="模型详情", padding="10")
        detail_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.detail_text = scrolledtext.ScrolledText(detail_frame, height=10, wrap=tk.WORD)
        self.detail_text.pack(fill=tk.BOTH, expand=True)
        
        self.results_tree.bind('<<TreeviewSelect>>', self.on_result_selected)
        self.refresh_results_list()
    
    def setup_predict_tab(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧：图像选择
        left_frame = ttk.Frame(frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5)
        
        # 模型加载
        load_frame = ttk.LabelFrame(left_frame, text="加载模型", padding="10")
        load_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(load_frame, text="从文件加载", command=self.load_model_from_file).pack(fill=tk.X, pady=5)
        ttk.Button(load_frame, text="从结果目录加载", command=self.load_model_from_results).pack(fill=tk.X, pady=5)
        
        # 图像选择
        image_frame = ttk.LabelFrame(left_frame, text="选择图像", padding="10")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Button(image_frame, text="选择图像文件", command=self.select_image).pack(pady=5)
        
        self.image_label = ttk.Label(image_frame, text="请选择一张图像")
        self.image_label.pack(pady=10, fill=tk.BOTH, expand=True)
        
        ttk.Button(image_frame, text="进行预测", command=self.predict_image).pack(pady=5)
        
        # 右侧：预测结果
        right_frame = ttk.Frame(frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        result_frame = ttk.LabelFrame(right_frame, text="预测结果", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.predict_result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD)
        self.predict_result_text.pack(fill=tk.BOTH, expand=True)
    
    def log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def browse_data_path(self):
        path = filedialog.askdirectory(title="选择数据集目录", initialdir=str(PROJECT_ROOT / 'data'))
        if path:
            self.data_path_var.set(path)
    
    def browse_results_path(self):
        path = filedialog.askdirectory(title="选择结果目录", initialdir=str(PROJECT_ROOT))
        if path:
            self.results_path_var.set(path)
            self.refresh_results_list()
    
    def load_dataset(self):
        try:
            data_path = Path(self.data_path_var.get())
            if not data_path.exists():
                messagebox.showerror("错误", f"数据集路径不存在: {data_path}")
                return
            
            # 使用与现有架构相同的数据加载逻辑
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            full_dataset = datasets.ImageFolder(root=str(data_path), transform=transform)
            self.class_names = full_dataset.classes
            
            # 划分数据集
            train_size = int(0.7 * len(full_dataset))
            val_size = int(0.15 * len(full_dataset))
            test_size = len(full_dataset) - train_size - val_size
            
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            info = f"数据集加载成功！\n\n"
            info += f"路径: {data_path}\n"
            info += f"总样本数: {len(full_dataset)}\n"
            info += f"训练集: {len(self.train_dataset)}\n"
            info += f"验证集: {len(self.val_dataset)}\n"
            info += f"测试集: {len(self.test_dataset)}\n"
            info += f"\n类别 ({len(self.class_names)}):\n"
            for i, name in enumerate(self.class_names):
                info += f"  {i}: {name}\n"
            
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(tk.END, info)
            self.log("数据集加载完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载数据集失败: {str(e)}")
            self.log(f"加载数据集失败: {str(e)}")
    
    def on_model_selected(self, event):
        model_name = self.model_var.get()
        if model_name in DEFAULT_PARAM_SPACE:
            params = DEFAULT_PARAM_SPACE[model_name]
            self.batch_size_var.set(params['batch_size'][0])
            self.lr_var.set(params['learning_rate'][0])
            self.weight_decay_var.set(params['weight_decay'][0])
            self.log(f"已加载 {model_name} 的默认参数")
    
    def start_training_thread(self):
        if self.is_training:
            return
        
        if self.train_dataset is None:
            messagebox.showwarning("警告", "请先加载数据集！")
            return
        
        self.is_training = True
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        thread = threading.Thread(target=self.train_model)
        thread.daemon = True
        thread.start()
    
    def stop_training(self):
        self.is_training = False
    
    def train_model(self):
        try:
            model_name = self.model_var.get()
            batch_size = self.batch_size_var.get()
            lr = self.lr_var.get()
            weight_decay = self.weight_decay_var.get()
            epochs = self.epochs_var.get()
            
            self.log(f"开始训练 {model_name}...")
            self.status_label.config(text="训练中...")
            
            # 创建数据加载器
            train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            # 创建模型
            model_class = MODEL_MAP[model_name]
            self.current_model = model_class(num_classes=len(self.class_names)).to(self.device)
            
            # 损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(self.current_model.parameters(), lr=lr, weight_decay=weight_decay)
            
            # 训练历史
            self.training_history = {
                'train_losses': [],
                'val_losses': [],
                'train_accs': [],
                'val_accs': []
            }
            
            best_val_acc = 0.0
            best_model_state = None
            
            for epoch in range(epochs):
                if not self.is_training:
                    self.log("训练已停止")
                    break
                
                # 训练阶段
                self.current_model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for data, target in train_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = self.current_model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    train_total += target.size(0)
                    train_correct += (predicted == target).sum().item()
                
                train_acc = 100.0 * train_correct / train_total
                avg_train_loss = train_loss / len(train_loader)
                
                # 验证阶段
                self.current_model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = self.current_model(data)
                        val_loss += criterion(output, target).item()
                        _, predicted = torch.max(output.data, 1)
                        val_total += target.size(0)
                        val_correct += (predicted == target).sum().item()
                
                val_acc = 100.0 * val_correct / val_total
                avg_val_loss = val_loss / len(val_loader)
                
                # 保存历史
                self.training_history['train_losses'].append(avg_train_loss)
                self.training_history['val_losses'].append(avg_val_loss)
                self.training_history['train_accs'].append(train_acc)
                self.training_history['val_accs'].append(val_acc)
                
                # 更新最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.current_model.state_dict().copy()
                
                # 更新UI
                progress = ((epoch + 1) / epochs) * 100
                self.progress_var.set(progress)
                
                history_line = f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%"
                self.history_text.insert(tk.END, history_line + "\n")
                self.history_text.see(tk.END)
                self.log(history_line)
            
            # 加载最佳模型
            if best_model_state is not None:
                self.current_model.load_state_dict(best_model_state)
            
            # 评估测试集
            test_acc = self.evaluate_test_set()
            
            # 绘制训练曲线
            self.plot_training_curves()
            
            self.status_label.config(text="训练完成")
            self.log(f"训练完成！最佳验证准确率: {best_val_acc:.2f}%, 测试集准确率: {test_acc:.2f}%")
            messagebox.showinfo("完成", f"训练完成！\n最佳验证准确率: {best_val_acc:.2f}%\n测试集准确率: {test_acc:.2f}%")
            
        except Exception as e:
            self.log(f"训练失败: {str(e)}")
            messagebox.showerror("错误", f"训练失败: {str(e)}")
        finally:
            self.is_training = False
            self.train_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
    
    def evaluate_test_set(self):
        if self.current_model is None or self.test_dataset is None:
            return 0.0
        
        test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        self.current_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.current_model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return 100.0 * correct / total
    
    def plot_training_curves(self):
        if self.training_history is None:
            return
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].plot(self.training_history['train_losses'], label='Train Loss')
            axes[0].plot(self.training_history['val_losses'], label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Loss Curves')
            axes[0].legend()
            axes[0].grid(True)
            
            axes[1].plot(self.training_history['train_accs'], label='Train Accuracy')
            axes[1].plot(self.training_history['val_accs'], label='Val Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_title('Accuracy Curves')
            axes[1].legend()
            axes[1].grid(True)
            
            plt.tight_layout()
            
            # 保存并显示
            plot_path = PROJECT_ROOT / 'temp_training_plot.png'
            plt.savefig(str(plot_path))
            plt.close()
            
            img = Image.open(str(plot_path))
            img.thumbnail((600, 400))
            photo = ImageTk.PhotoImage(img)
            self.plot_label.config(image=photo, text="")
            self.plot_label.image = photo
            
        except Exception as e:
            self.log(f"绘制图表失败: {str(e)}")
    
    def save_model(self):
        if self.current_model is None:
            messagebox.showwarning("警告", "没有可保存的模型！")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pt",
                filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")],
                initialdir=str(PROJECT_ROOT)
            )
            if file_path:
                save_dict = {
                    'model_state_dict': self.current_model.state_dict(),
                    'model_name': self.model_var.get(),
                    'class_names': self.class_names,
                    'training_history': self.training_history
                }
                torch.save(save_dict, file_path)
                messagebox.showinfo("成功", f"模型已保存到: {file_path}")
                self.log(f"模型已保存: {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存模型失败: {str(e)}")
            self.log(f"保存模型失败: {str(e)}")
    
    def refresh_results_list(self):
        results_path = Path(self.results_path_var.get())
        if not results_path.exists():
            return
        
        # 清空列表
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # 扫描结果目录
        for model_dir in results_path.iterdir():
            if model_dir.is_dir():
                results_file = model_dir / 'model_results.json'
                if results_file.exists():
                    try:
                        with open(results_file, 'r', encoding='utf-8') as f:
                            results = json.load(f)
                        
                        self.results_tree.insert('', 'end', values=(
                            results['model_name'],
                            f"{results['mean_acc']:.2f}%",
                            f"{results['std_acc']:.2f}%",
                            f"{results['max_acc']:.2f}%",
                            f"{results['min_acc']:.2f}%"
                        ))
                    except Exception as e:
                        continue
    
    def on_result_selected(self, event):
        selection = self.results_tree.selection()
        if not selection:
            return
        
        item = self.results_tree.item(selection[0])
        model_name = item['values'][0]
        
        results_path = Path(self.results_path_var.get())
        model_dir = results_path / model_name
        results_file = model_dir / 'model_results.json'
        
        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                detail = f"模型: {results['model_name']}\n\n"
                detail += f"最佳参数: {results['best_params']}\n\n"
                detail += "各种子结果:\n"
                for seed, seed_result in results['seed_results'].items():
                    detail += f"  Seed {seed}: {seed_result['best_val_acc']:.2f}%\n"
                
                self.detail_text.delete(1.0, tk.END)
                self.detail_text.insert(tk.END, detail)
                
                # 显示训练曲线
                plot_file = model_dir / 'model_comparison.png'
                if plot_file.exists():
                    try:
                        img = Image.open(str(plot_file))
                        img.thumbnail((400, 300))
                        # 这里可以添加显示逻辑
                    except:
                        pass
                        
            except Exception as e:
                self.detail_text.delete(1.0, tk.END)
                self.detail_text.insert(tk.END, f"加载详情失败: {str(e)}")
    
    def load_model_from_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")],
            initialdir=str(PROJECT_ROOT)
        )
        if file_path:
            self.load_model(file_path)
    
    def load_model_from_results(self):
        """从结果目录加载模型，允许用户选择具体的模型和种子"""
        results_path = Path(self.results_path_var.get())
        
        if not results_path.exists():
            messagebox.showerror("错误", f"结果目录不存在: {results_path}")
            return
        
        model_dirs = [d for d in results_path.iterdir() if d.is_dir()]
        
        if not model_dirs:
            messagebox.showwarning("警告", "结果目录中没有找到模型！")
            return
        
        # 创建选择对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("选择要加载的模型")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="请选择模型和种子:", padding=10).pack(anchor=tk.W)
        
        # 创建Treeview显示可用模型
        frame = ttk.Frame(dialog, padding=5)
        frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('model', 'seed', 'accuracy', 'file')
        tree = ttk.Treeview(frame, columns=columns, show='headings', selectmode='browse')
        tree.heading('model', text='模型名称')
        tree.heading('seed', text='随机种子')
        tree.heading('accuracy', text='验证准确率')
        tree.heading('file', text='文件路径')
        
        tree.column('model', width=120)
        tree.column('seed', width=80)
        tree.column('accuracy', width=100)
        tree.column('file', width=200)
        
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 填充可用模型列表
        available_models = []
        for model_dir in sorted(model_dirs):
            model_name = model_dir.name
            if model_name not in MODEL_MAP:
                continue
                
            # 尝试读取准确率信息
            results_file = model_dir / 'model_results.json'
            acc_info = {}
            if results_file.exists():
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    acc_info = {k: v['best_val_acc'] for k, v in results.get('seed_results', {}).items()}
                except:
                    pass
            
            for seed in [45, 46, 47]:
                model_file = model_dir / f'best_model_seed{seed}.pt'
                if model_file.exists():
                    acc = acc_info.get(str(seed), 'N/A')
                    if isinstance(acc, float):
                        acc = f"{acc:.2f}%"
                    item_id = tree.insert('', 'end', values=(
                        model_name,
                        str(seed),
                        acc,
                        str(model_file)
                    ))
                    available_models.append((item_id, str(model_file), model_name))
        
        def on_select_and_load():
            selection = tree.selection()
            if not selection:
                messagebox.showwarning("警告", "请先选择一个模型！")
                return
            
            item_id = selection[0]
            # 找到对应的文件路径
            for (tid, file_path, model_name) in available_models:
                if tid == item_id:
                    dialog.destroy()
                    # 先设置模型类型
                    self.model_var.set(model_name)
                    # 再加载模型
                    self.load_model(file_path, model_name)
                    return
        
        btn_frame = ttk.Frame(dialog, padding=10)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="加载选中模型", command=on_select_and_load).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # 双击也可以加载
        tree.bind('<Double-1>', lambda e: on_select_and_load())
    
    def extract_model_name_from_path(self, file_path):
        """从文件路径推断模型名称"""
        path = Path(file_path)
        
        # 方法1：从父目录名推断（如 proper_grid_results/ResNet18/best_model_seed45.pt）
        parent_dir = path.parent.name
        if parent_dir in MODEL_MAP:
            return parent_dir
        
        # 方法2：从文件名推断
        stem = path.stem.lower()
        for model_name in MODEL_MAP.keys():
            if model_name.lower() in stem:
                return model_name
        
        # 方法3：从更上层的目录结构推断
        for parent in path.parents:
            if parent.name in MODEL_MAP:
                return parent.name
        
        return None
    
    def load_model(self, file_path, model_name=None):
        """
        加载模型文件
        :param file_path: 模型文件路径
        :param model_name: 可选的模型名称，如果为None则尝试自动推断
        """
        try:
            checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
            
            # 判断模型格式并确定模型名称
            if 'model_name' in checkpoint:
                # 格式1：GUI保存的格式（包含model_name字段）
                detected_model_name = checkpoint['model_name']
                self.class_names = checkpoint.get('class_names')
                self.training_history = checkpoint.get('training_history')
            elif model_name:
                # 格式2：调用者提供了模型名称
                detected_model_name = model_name
            else:
                # 格式3：需要从路径推断（现有实验结果格式）
                detected_model_name = self.extract_model_name_from_path(file_path)
                
                if not detected_model_name:
                    # 无法推断，询问用户
                    dialog = tk.Toplevel(self.root)
                    dialog.title("选择模型类型")
                    dialog.geometry("300x200")
                    dialog.transient(self.root)
                    dialog.grab_set()
                    
                    ttk.Label(dialog, text="无法自动识别模型类型\n请手动选择:", padding=10).pack()
                    
                    model_var = tk.StringVar()
                    combo = ttk.Combobox(dialog, textvariable=model_var, 
                                        values=list(MODEL_MAP.keys()), state='readonly')
                    combo.pack(padx=20, pady=10)
                    combo.set('ResNet18')  # 默认值
                    
                    result = [None]
                    
                    def on_confirm():
                        result[0] = model_var.get()
                        dialog.destroy()
                    
                    ttk.Button(dialog, text="确定", command=on_confirm).pack(pady=10)
                    dialog.wait_window()
                    detected_model_name = result[0]
            
            # 验证模型名称是否有效
            if detected_model_name not in MODEL_MAP:
                raise ValueError(f"未知的模型类型: {detected_model_name}")
            
            # 创建模型实例
            model_class = MODEL_MAP[detected_model_name]
            
            # 确定类别数量
            num_classes = 10  # 默认值
            if self.class_names:
                num_classes = len(self.class_names)
            elif 'params' in checkpoint:
                # 尝试从params中获取信息（虽然通常没有num_classes）
                pass
            
            # 创建模型并加载权重
            self.current_model = model_class(num_classes=num_classes).to(self.device)
            self.current_model.load_state_dict(checkpoint['model_state_dict'])
            self.current_model.eval()
            
            # 更新UI状态
            self.model_var.set(detected_model_name)
            
            # 获取额外信息用于日志
            extra_info = []
            if 'best_val_acc' in checkpoint:
                extra_info.append(f"验证准确率: {checkpoint['best_val_acc']:.2f}%")
            if 'epoch' in checkpoint:
                extra_info.append(f"训练轮数: {checkpoint['epoch']}")
            if 'seed' in checkpoint:
                extra_info.append(f"随机种子: {checkpoint['seed']}")
            
            info_str = f" ({', '.join(extra_info)})" if extra_info else ""
            
            messagebox.showinfo("成功", f"模型已成功加载！\n\n模型类型: {detected_model_name}\n类别数量: {num_classes}{info_str}")
            self.log(f"✓ 模型已加载: {Path(file_path).name} [{detected_model_name}]")
            if extra_info:
                self.log(f"  模型信息: {'; '.join(extra_info)}")
            
        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}"
            messagebox.showerror("错误", error_msg)
            self.log(f"✗ {error_msg}")
            import traceback
            self.log(f"详细错误:\n{traceback.format_exc()}")
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All Files", "*.*")],
            initialdir=str(PROJECT_ROOT / 'data')
        )
        if file_path:
            self.current_image_path = file_path
            img = Image.open(file_path)
            img.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            self.log(f"已选择图像: {file_path}")
    
    def predict_image(self):
        if self.current_model is None:
            messagebox.showwarning("警告", "请先加载模型！")
            return
        
        if not hasattr(self, 'current_image_path'):
            messagebox.showwarning("警告", "请先选择图像！")
            return
        
        try:
            # 判断模型需要的输入尺寸
            model_name = self.model_var.get()
            input_size = (299, 299) if model_name == 'InceptionV3' else (224, 224)
            
            # 图像预处理
            transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            img = Image.open(self.current_image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            # 预测
            self.current_model.eval()
            with torch.no_grad():
                output = self.current_model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = torch.max(output.data, 1)
            
            predicted_class = predicted.item()
            confidence = probabilities[0][predicted_class].item() * 100
            
            result = f"预测结果\n\n"
            result += f"预测类别: {predicted_class}"
            if self.class_names:
                result += f" ({self.class_names[predicted_class]})\n"
            result += f"\n置信度: {confidence:.2f}%\n\n"
            result += "各类别概率:\n"
            
            for i in range(len(probabilities[0])):
                prob = probabilities[0][i].item() * 100
                class_name = self.class_names[i] if self.class_names else f"Class {i}"
                result += f"  {class_name}: {prob:.2f}%\n"
            
            self.predict_result_text.delete(1.0, tk.END)
            self.predict_result_text.insert(tk.END, result)
            self.log(f"预测完成: 类别={predicted_class}, 置信度={confidence:.2f}%")
            
        except Exception as e:
            messagebox.showerror("错误", f"预测失败: {str(e)}")
            self.log(f"预测失败: {str(e)}")


def main():
    root = tk.Tk()
    app = CNNComparisonGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
