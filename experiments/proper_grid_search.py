#!/usr/bin/env python
"""
正确的网格搜索系统
按照真正的网格搜索逻辑实现：
1. 阶段1：对所有参数组合进行快速评估（10轮，种子=42）
2. 找到最佳参数组合
3. 阶段2：对最佳参数组合进行200轮余弦退火训练（种子45,46,47）
"""

import os
import sys
import time
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')

# 添加当前目录到路径以便导入模型
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# 导入所有经典CNN模型
from models.LeNet_5 import LeNet5
from models.AlexNet import AlexNet
from models.VGG import VGG11, VGG13, VGG16, VGG19
from models.ResNet import ResNet18, ResNet34, ResNet50
from models.Inception import InceptionV1, InceptionV3
from models.DenseNet import DenseNet121, DenseNet169

# 导入可视化库
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def setup_logging(log_dir='logs'):
    """
    设置日志系统
    
    参数:
        log_dir: 日志目录
        
    返回:
        logger: 配置好的日志记录器
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'proper_grid_search_{timestamp}.log')
    
    # 配置日志记录器
    logger = logging.getLogger('ProperGridSearch')
    logger.setLevel(logging.DEBUG)
    
    # 防止重复添加handler
    if logger.handlers:
        logger.handlers.clear()
    
    # 文件handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

def clean_working_directory(base_dir=None, logger=None):
    """
    清理工作目录
    
    参数:
        base_dir: 基础目录
        logger: 日志记录器
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent
    
    directories_to_clean = [
        'new_grid_results',
        'bayesian_results',
        'optimized_grid_results',
        'checkpoints',
        'results',
        'reports',
        'logs',
        'tensorboard_logs',
        'final_results',
        'proper_grid_results'
    ]
    
    if logger:
        logger.info("开始清理工作目录...")
    
    # 清理目录
    for dir_name in directories_to_clean:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            try:
                import shutil
                shutil.rmtree(dir_path)
                if logger:
                    logger.info(f"  已删除目录: {dir_name}")
            except Exception as e:
                if logger:
                    logger.warning(f"  删除目录失败 {dir_name}: {e}")
    
    # 创建新的必要目录
    os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'proper_grid_results'), exist_ok=True)
    
    if logger:
        logger.info("工作目录清理完成")

class ProperGridSearch:
    """
    正确的网格搜索类
    按照真正的网格搜索逻辑实现：
    阶段1：对所有参数组合进行快速评估（10轮，种子=42）
    找到最佳参数组合
    阶段2：对最佳参数组合进行200轮余弦退火训练（种子45,46,47）
    """
    
    def __init__(self, model_set, param_space, output_dir='proper_grid_results', logger=None, resume=True):
        """
        初始化正确的网格搜索
        
        参数:
            model_set: 要评估的模型列表
            param_space: 每个模型的参数搜索空间
            output_dir: 输出目录
            logger: 日志记录器
            resume: 是否启用断点续传
        """
        self.model_set = model_set
        self.param_space = param_space
        self.output_dir = output_dir
        self.logger = logger
        self.resume = resume
        
        self._dataset_cache = {}
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设备选择和GPU加速优化
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 启用CuDNN基准（自动选择最快的卷积算法）
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # 模型类映射
        self._model_class_map = {
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
        
        if self.logger:
            self.logger.info("初始化正确的网格搜索")
            self.logger.info(f"  模型集合: {model_set}")
            self.logger.info(f"  设备: {self.device}")
    
    def _set_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        if self.logger:
            self.logger.info(f"  设置随机种子: {seed}")
    
    def _load_dataset(self, model_name):
        """加载数据集，使用缓存提高效率"""
        if model_name in self._dataset_cache:
            return self._dataset_cache[model_name]
        
        # 构建数据路径
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
            'ishihara_dataset',
            'train'
        )
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据路径不存在: {data_path}")
        
        # 根据模型选择输入尺寸
        if model_name == 'InceptionV3':
            input_size = (299, 299)
        else:
            input_size = (224, 224)
        
        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 加载数据集
        full_dataset = datasets.ImageFolder(root=data_path, transform=transform)
        
        # 划分训练集和验证集 (80%训练, 20%验证)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        if self.logger:
            self.logger.info(f"数据集加载成功: {model_name}")
            self.logger.info(f"  训练集: {len(train_dataset)} 样本")
            self.logger.info(f"  验证集: {len(val_dataset)} 样本")
        
        # 缓存数据集
        self._dataset_cache[model_name] = (train_dataset, val_dataset)
        
        return train_dataset, val_dataset
    
    def _train_quick_evaluation(self, model_name, params, seed, train_dataset, val_dataset, model_dir):
        """
        阶段1：10轮快速评估
        
        参数:
            model_name: 模型名称
            params: 参数字典
            seed: 随机种子
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            model_dir: 模型保存目录
            
        返回:
            最佳验证准确率和训练历史
        """
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']
        
        if self.logger:
            self.logger.info(f"\n  参数组合: {params}")
            self.logger.info(f"  快速评估: 10轮")
        
        # 设置种子
        self._set_seed(seed)
        
        # 创建数据加载器（添加加速优化）
        num_workers = 4 if torch.cuda.is_available() else 0
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0)
        )
        
        # 获取模型类
        model_class = self._model_class_map[model_name]
        num_classes = len(train_dataset.dataset.classes)
        
        # 创建模型实例
        model = model_class(num_classes=num_classes)
        model = model.to(self.device)
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        
        # 记录训练历史
        best_val_acc = 0.0
        val_accs = []
        
        for epoch in range(10):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                    output = model(data)
                    loss = criterion(output, target)
                
                scaler.scale(loss).backward()
                
                if self.device.type == 'cuda':
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            train_acc = 100.0 * train_correct / train_total
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                        output = model(data)
                        val_loss += criterion(output, target).item()
                    
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            val_acc = 100.0 * val_correct / val_total
            val_accs.append(val_acc)
            
            if self.logger:
                self.logger.info(f"    Epoch {epoch+1}/10: 训练={train_acc:.2f}%, 验证={val_acc:.2f}%")
            
            # 更新最佳准确率
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        training_history = {
            'val_accs': val_accs,
            'best_val_acc': best_val_acc
        }
        
        return best_val_acc, training_history
    
    def _train_cosine_annealing(self, model_name, params, seed, train_dataset, val_dataset, model_dir):
        """
        阶段2：200轮余弦退火训练
        
        参数:
            model_name: 模型名称
            params: 参数字典
            seed: 随机种子
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            model_dir: 模型保存目录
            
        返回:
            最佳验证准确率和训练历史
        """
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']
        
        if self.logger:
            self.logger.info(f"\n  最佳参数组合: {params}")
            self.logger.info(f"  Seed: {seed}")
            self.logger.info(f"  余弦退火训练: 200轮，耐心=15")
        
        # 设置种子
        self._set_seed(seed)
        
        # 创建数据加载器（添加加速优化）
        num_workers = 4 if torch.cuda.is_available() else 0
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0)
        )
        
        # 获取模型类
        model_class = self._model_class_map[model_name]
        num_classes = len(train_dataset.dataset.classes)
        
        # 创建模型实例
        model = model_class(num_classes=num_classes)
        model = model.to(self.device)
        
        # 创建TensorBoard writer
        writer_dir = os.path.join(model_dir, f'tensorboard_seed{seed}')
        os.makedirs(writer_dir, exist_ok=True)
        writer = SummaryWriter(writer_dir)
        
        # 损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 余弦退火学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=200,
            eta_min=1e-6
        )
        
        # 早停配置
        patience = 15
        min_delta = 0.01
        patience_counter = 0
        best_val_acc = 0.0
        best_model_state = None
        
        # 记录训练历史
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        learning_rates = []
        
        # 混合精度训练
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        
        for epoch in range(200):
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                    output = model(data)
                    loss = criterion(output, target)
                
                scaler.scale(loss).backward()
                
                if self.device.type == 'cuda':
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            train_acc = 100.0 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                        output = model(data)
                        val_loss += criterion(output, target).item()
                    
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            val_acc = 100.0 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            # 更新学习率
            scheduler.step()
            
            # 记录历史
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # TensorBoard记录
            writer.add_scalar('Final/Train_Loss', avg_train_loss, epoch)
            writer.add_scalar('Final/Val_Loss', avg_val_loss, epoch)
            writer.add_scalar('Final/Train_Acc', train_acc, epoch)
            writer.add_scalar('Final/Val_Acc', val_acc, epoch)
            writer.add_scalar('Final/Learning_Rate', current_lr, epoch)
            
            if self.logger:
                self.logger.info(f"    Epoch {epoch+1}/200: 训练={train_acc:.2f}%, 验证={val_acc:.2f}%, LR={current_lr:.6f}")
            
            # 早停检查
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                
                # 保存最佳模型
                torch.save({
                    'model_state_dict': best_model_state,
                    'epoch': epoch,
                    'best_val_acc': best_val_acc,
                    'params': params,
                    'seed': seed
                }, os.path.join(model_dir, f'best_model_seed{seed}.pt'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if self.logger:
                        self.logger.info(f"    早停触发: 连续{patience}个epoch未提升")
                    break
        
        # 关闭TensorBoard writer
        writer.close()
        
        # 保存训练历史
        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'learning_rates': learning_rates,
            'best_val_acc': best_val_acc
        }
        
        # 绘制训练曲线
        self._plot_training_curves(training_history, model_dir, model_name, seed)
        
        # 保存训练历史
        with open(os.path.join(model_dir, f'training_history_seed{seed}.json'), 'w', encoding='utf-8') as f:
            json.dump(training_history, f, indent=4, default=str)
        
        # 保存为CSV
        training_df = pd.DataFrame({
            'epoch': list(range(1, len(train_accs) + 1)),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accs,
            'val_acc': val_accs,
            'learning_rate': learning_rates
        })
        training_df.to_csv(os.path.join(model_dir, f'training_results_seed{seed}.csv'), index=False)
        
        return best_val_acc, training_history
    
    def _plot_training_curves(self, training_history, model_dir, model_name, seed):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(training_history['train_losses'], label='Train Loss')
        axes[0, 0].plot(training_history['val_losses'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title(f'{model_name} (Seed={seed}) - Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(training_history['train_accs'], label='Train Acc')
        axes[0, 1].plot(training_history['val_accs'], label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title(f'{model_name} (Seed={seed}) - Accuracy Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率曲线
        axes[1, 0].plot(training_history['learning_rates'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule (Cosine Annealing)')
        axes[1, 0].grid(True)
        
        # 准确率分布
        axes[1, 1].hist(training_history['val_accs'], bins=20, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(training_history['best_val_acc'], color='red', linestyle='--', 
                          label=f'Best: {training_history["best_val_acc"]:.2f}%')
        axes[1, 1].set_xlabel('Validation Accuracy (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Validation Accuracy Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f'training_curves_seed{seed}.png'))
        plt.close()
    
    def _check_phase1_completed(self, model_name):
        """检查阶段1是否完成"""
        model_dir = os.path.join(self.output_dir, model_name)
        phase1_file = os.path.join(model_dir, 'phase1_results.json')
        return os.path.exists(phase1_file)
    
    def _load_phase1_results(self, model_name):
        """加载阶段1结果"""
        model_dir = os.path.join(self.output_dir, model_name)
        phase1_file = os.path.join(model_dir, 'phase1_results.json')
        if os.path.exists(phase1_file):
            with open(phase1_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def run(self):
        """
        执行正确的网格搜索
        
        返回:
            完整的实验结果
        """
        start_time = time.time()
        experiment_start_time = datetime.now()
        
        if self.logger:
            self.logger.info("=" * 80)
            self.logger.info("开始正确的网格搜索实验")
            self.logger.info(f"开始时间: {experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("=" * 80)
        
        # 存储所有结果
        all_results = {}
        
        # 逐个模型进行评估
        for model_name in self.model_set:
            if self.logger:
                self.logger.info(f"\n{'=' * 80}")
                self.logger.info(f"开始评估模型: {model_name}")
                self.logger.info(f"{'=' * 80}")
            
            # 创建模型目录
            model_dir = os.path.join(self.output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # 加载数据集
            train_dataset, val_dataset = self._load_dataset(model_name)
            
            # 检查阶段1是否已完成
            phase1_completed = self.resume and self._check_phase1_completed(model_name)
            
            phase1_results = None
            best_params = None
            
            if phase1_completed:
                if self.logger:
                    self.logger.info("✓ 阶段1已完成，加载已有结果")
                phase1_results = self._load_phase1_results(model_name)
                best_params = phase1_results['best_params']
            else:
                # ========== 阶段1：快速评估所有参数组合 ==========
                if self.logger:
                    self.logger.info("\n--- 阶段1：快速评估所有参数组合 ---")
                
                param_space = self.param_space.get(model_name, {
                    'batch_size': [16, 32, 64],
                    'learning_rate': [0.0001, 0.001, 0.002],
                    'weight_decay': [0.0, 0.0001, 0.001]
                })
                
                # 生成所有参数组合
                param_combinations = []
                for batch_size in param_space.get('batch_size', [32]):
                    for learning_rate in param_space.get('learning_rate', [0.001]):
                        for weight_decay in param_space.get('weight_decay', [0.0001]):
                            param_combinations.append({
                                'batch_size': batch_size,
                                'learning_rate': learning_rate,
                                'weight_decay': weight_decay
                            })
                
                if self.logger:
                    self.logger.info(f"参数组合数量: {len(param_combinations)}")
                
                # 评估每个参数组合
                param_results = []
                for idx, params in enumerate(param_combinations):
                    if self.logger:
                        self.logger.info(f"\n参数组合 {idx+1}/{len(param_combinations)}")
                    
                    # 快速评估（10轮，种子=42）
                    best_acc, history = self._train_quick_evaluation(
                        model_name, params, 42,
                        train_dataset, val_dataset,
                        model_dir
                    )
                    
                    param_results.append({
                        'params': params,
                        'best_val_acc': best_acc,
                        'history': history
                    })
                
                # 找到最佳参数组合
                best_idx = np.argmax([r['best_val_acc'] for r in param_results])
                best_params = param_results[best_idx]['params']
                best_acc = param_results[best_idx]['best_val_acc']
                
                if self.logger:
                    self.logger.info(f"\n阶段1完成！")
                    self.logger.info(f"最佳参数组合: {best_params}")
                    self.logger.info(f"最佳准确率: {best_acc:.2f}%")
                
                # 保存阶段1结果
                phase1_results = {
                    'param_combinations': param_combinations,
                    'param_results': [
                        {'params': r['params'], 'best_val_acc': r['best_val_acc']}
                        for r in param_results
                    ],
                    'best_params': best_params,
                    'best_val_acc': best_acc
                }
                
                with open(os.path.join(model_dir, 'phase1_results.json'), 'w', encoding='utf-8') as f:
                    json.dump(phase1_results, f, indent=4, default=str)
            
            # ========== 阶段2：最佳参数组合的余弦退火训练 ==========
            if self.logger:
                self.logger.info(f"\n--- 阶段2：最佳参数组合的余弦退火训练 ---")
            
            seeds = [45, 46, 47]
            seed_results = {}
            
            for seed in seeds:
                # 检查是否已完成
                if self.resume:
                    check_files = [
                        os.path.join(model_dir, f'best_model_seed{seed}.pt'),
                        os.path.join(model_dir, f'training_history_seed{seed}.json')
                    ]
                    if all(os.path.exists(f) for f in check_files):
                        if self.logger:
                            self.logger.info(f"✓ Seed {seed} 已完成，跳过")
                        
                        # 加载已有结果
                        with open(os.path.join(model_dir, f'training_history_seed{seed}.json'), 'r', encoding='utf-8') as f:
                            history = json.load(f)
                        seed_results[seed] = {
                            'best_val_acc': history['best_val_acc']
                        }
                        continue
                
                # 余弦退火训练
                final_acc, training_history = self._train_cosine_annealing(
                    model_name, best_params, seed,
                    train_dataset, val_dataset,
                    model_dir
                )
                
                seed_results[seed] = {
                    'best_val_acc': final_acc
                }
            
            # 计算统计结果
            seed_accs = [seed_results[seed]['best_val_acc'] for seed in seeds]
            mean_acc = np.mean(seed_accs)
            std_acc = np.std(seed_accs)
            max_acc = np.max(seed_accs)
            min_acc = np.min(seed_accs)
            
            # 保存模型结果
            model_result = {
                'model_name': model_name,
                'best_params': best_params,
                'seed_results': seed_results,
                'mean_acc': mean_acc,
                'std_acc': std_acc,
                'max_acc': max_acc,
                'min_acc': min_acc,
                'phase1_results': phase1_results
            }
            
            all_results[model_name] = model_result
            
            with open(os.path.join(model_dir, 'model_results.json'), 'w', encoding='utf-8') as f:
                json.dump(model_result, f, indent=4, default=str)
            
            # 绘制模型对比图表
            self._plot_model_comparison(model_result, model_dir, model_name)
            
            if self.logger:
                self.logger.info(f"\n{model_name} 评估完成")
                self.logger.info(f"  平均准确率: {mean_acc:.2f}% (±{std_acc:.2f}%)")
                self.logger.info(f"  最高准确率: {max_acc:.2f}%")
                self.logger.info(f"  最低准确率: {min_acc:.2f}%")
        
        # 保存完整结果
        experiment_end_time = datetime.now()
        final_report = {
            'experiment_start_time': experiment_start_time.isoformat(),
            'experiment_end_time': experiment_end_time.isoformat(),
            'total_duration': (experiment_end_time - experiment_start_time).total_seconds(),
            'model_set': self.model_set,
            'param_space': self.param_space,
            'all_results': all_results
        }
        
        with open(os.path.join(self.output_dir, 'final_experiment_report.json'), 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=4, default=str)
        
        # 生成最终报告
        self._generate_final_report(all_results, start_time, experiment_start_time, experiment_end_time)
        
        if self.logger:
            total_duration = (experiment_end_time - experiment_start_time).total_seconds() / 3600
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info("正确的网格搜索完成！")
            self.logger.info(f"结束时间: {experiment_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"总耗时: {total_duration:.2f}小时")
            self.logger.info("=" * 80)
        
        return all_results
    
    def _plot_model_comparison(self, model_result, model_dir, model_name):
        """绘制模型对比图表"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 不同种子的准确率对比
        seeds = list(model_result['seed_results'].keys())
        seed_accs = [model_result['seed_results'][seed]['best_val_acc'] for seed in seeds]
        
        bars = axes[0].bar(range(len(seeds)), seed_accs, color='steelblue')
        axes[0].set_xlabel('Seed')
        axes[0].set_ylabel('Best Validation Accuracy (%)')
        axes[0].set_title(f'{model_name} - Seed Comparison')
        axes[0].set_xticks(range(len(seeds)))
        axes[0].set_xticklabels([f'Seed {s}' for s in seeds])
        axes[0].grid(True, axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, acc in zip(bars, seed_accs):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{acc:.2f}%',
                        ha='center', va='bottom')
        
        # 统计信息
        stats_text = f"Mean: {model_result['mean_acc']:.2f}%\n"
        stats_text += f"Std: {model_result['std_acc']:.2f}%\n"
        stats_text += f"Max: {model_result['max_acc']:.2f}%\n"
        stats_text += f"Min: {model_result['min_acc']:.2f}%"
        
        axes[1].text(0.1, 0.5, stats_text, 
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].axis('off')
        axes[1].set_title('Statistics')
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'model_comparison.png'))
        plt.close()
    
    def _generate_final_report(self, all_results, start_time, 
                               experiment_start_time, experiment_end_time):
        """生成最终报告"""
        report_path = os.path.join(self.output_dir, 'FINAL_EXPERIMENT_REPORT.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 最终实验报告（正确的网格搜索）\n\n")
            f.write(f"**开始时间**: {experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**结束时间**: {experiment_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            total_duration = (experiment_end_time - experiment_start_time).total_seconds() / 3600
            f.write(f"**总耗时**: {total_duration:.2f}小时\n\n")
            
            f.write("## 实验设计\n\n")
            f.write("### 阶段1：快速评估\n\n")
            f.write("- 对所有参数组合进行10轮快速评估\n")
            f.write("- 使用种子：42\n")
            f.write("- 找到最佳参数组合\n\n")
            
            f.write("### 阶段2：余弦退火训练\n\n")
            f.write("- 仅对最佳参数组合进行200轮训练\n")
            f.write("- 使用种子：45, 46, 47\n")
            f.write("- 学习率调度：CosineAnnealingLR\n")
            f.write("- 早停耐心：15\n\n")
            
            f.write("## 实验配置\n\n")
            f.write(f"- **模型集合**: {list(all_results.keys())}\n\n")
            
            f.write("## 实验结果\n\n")
            
            # 结果表格
            f.write("| 模型 | 平均准确率 | 标准差 | 最高准确率 | 最低准确率 |\n")
            f.write("|------|-----------|--------|-----------|-----------|\n")
            
            for model_name, model_data in all_results.items():
                f.write(f"| {model_name} | {model_data['mean_acc']:.2f}% | "
                       f"{model_data['std_acc']:.2f}% | "
                       f"{model_data['max_acc']:.2f}% | "
                       f"{model_data['min_acc']:.2f}% |\n")
            
            f.write("\n## 各模型详细结果\n\n")
            for model_name, model_data in all_results.items():
                f.write(f"### {model_name}\n\n")
                f.write(f"**最佳参数**: {model_data['best_params']}\n\n")
                f.write("| Seed | 最终准确率 |\n")
                f.write("|------|-----------|\n")
                
                seeds = [45, 46, 47]
                for seed in seeds:
                    acc = model_data['seed_results'][seed]['best_val_acc']
                    f.write(f"| {seed} | {acc:.2f}% |\n")
                f.write("\n")
        
        if self.logger:
            self.logger.info(f"最终报告已生成: {report_path}")
