#!/usr/bin/env python
"""
正确的网格搜索实验运行脚本
按照真正的网格搜索逻辑：
1. 阶段1：对所有参数组合进行快速评估（10轮，种子=42）
2. 找到最佳参数组合
3. 阶段2：对最佳参数组合进行200轮余弦退火训练（种子45,46,47）
"""

import os
import sys
from pathlib import Path

# 在导入其他模块前设置环境变量，禁用警告
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 禁用TensorFlow日志

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'experiments'))

from experiments.proper_grid_search import setup_logging, clean_working_directory, ProperGridSearch

def main():
    """主函数"""
    # 设置日志
    logger, log_file = setup_logging()
    
    logger.info("=" * 80)
    logger.info("正确的网格搜索实验")
    logger.info("=" * 80)
    
    try:
        # 清理工作目录（仅在首次运行时启用）
        # 注意：如果启用断点续传，不要清理工作目录
        clean_workspace = False  # 设置为False以启用断点续传
        
        if clean_workspace:
            logger.info("步骤1: 清理工作目录")
            clean_working_directory(logger=logger)
        else:
            logger.info("步骤1: 跳过工作目录清理（断点续传模式）")
        
        # 定义要评估的模型集合（可以修改这里）
        model_set = [
            'LeNet5',
            'ResNet18',
            'ResNet34',
            'ResNet50',
            'VGG11',
            'VGG13',
            'VGG16',
            'VGG19',
            'AlexNet',
            'InceptionV1',
            'InceptionV3',
            'DenseNet121',
            'DenseNet169'
        ]
        
        # 定义每个模型的参数搜索空间
        param_space = {
            'LeNet5': {
                'batch_size': [32, 64],
                'learning_rate': [0.001, 0.005],
                'weight_decay': [0.0, 0.0001]
            },
            'ResNet18': {
                'batch_size': [16, 32, 64],
                'learning_rate': [0.0001, 0.0005, 0.001],
                'weight_decay': [0.0, 0.00005, 0.0001]
            },
            'ResNet34': {
                'batch_size': [16, 32],
                'learning_rate': [0.0001, 0.0005],
                'weight_decay': [0.0, 0.0001]
            },
            'ResNet50': {
                'batch_size': [16, 32],
                'learning_rate': [0.0001, 0.0005],
                'weight_decay': [0.0, 0.0001]
            },
            'VGG11': {
                'batch_size': [16, 32],
                'learning_rate': [0.0005, 0.001],
                'weight_decay': [0.00005, 0.0001]
            },
            'VGG13': {
                'batch_size': [16, 32],
                'learning_rate': [0.0005, 0.001],
                'weight_decay': [0.00005, 0.0001]
            },
            'VGG16': {
                'batch_size': [16, 32],
                'learning_rate': [0.0005, 0.001, 0.002],
                'weight_decay': [0.00005, 0.0001]
            },
            'VGG19': {
                'batch_size': [16, 32],
                'learning_rate': [0.0005, 0.001],
                'weight_decay': [0.00005, 0.0001]
            },
            'AlexNet': {
                'batch_size': [16, 32, 64],
                'learning_rate': [0.0001, 0.0005, 0.001],
                'weight_decay': [0.0, 0.0001, 0.0005]
            },
            'InceptionV1': {
                'batch_size': [32, 64],
                'learning_rate': [0.001, 0.005],
                'weight_decay': [0.0, 0.0001]
            },
            'InceptionV3': {
                'batch_size': [16, 32],
                'learning_rate': [0.001, 0.005],
                'weight_decay': [0.0, 0.0001]
            },
            'DenseNet121': {
                'batch_size': [16, 32],
                'learning_rate': [0.001, 0.005],
                'weight_decay': [0.0, 0.0001]
            },
            'DenseNet169': {
                'batch_size': [16, 32],
                'learning_rate': [0.001, 0.005],
                'weight_decay': [0.0, 0.0001]
            }
        }
        
        # 打印实验配置
        logger.info("\n步骤2: 实验配置")
        logger.info(f"  模型集合: {model_set}")
        logger.info(f"  断点续传: 启用")
        
        # 创建正确的网格搜索器
        logger.info("初始化正确的网格搜索器...")
        optimizer = ProperGridSearch(
            model_set=model_set,
            param_space=param_space,
            output_dir='proper_grid_results',
            logger=logger,
            resume=True  # 启用断点续传
        )
        
        # 执行搜索
        logger.info("\n步骤3: 开始正确的网格搜索")
        logger.info("=" * 80)
        all_results = optimizer.run()
        logger.info("=" * 80)
        
        # 总结
        logger.info("\n" + "=" * 80)
        logger.info("正确的网格搜索完成！")
        logger.info(f"日志文件: {log_file}")
        logger.info("结果目录: proper_grid_results/")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"\n实验执行错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
