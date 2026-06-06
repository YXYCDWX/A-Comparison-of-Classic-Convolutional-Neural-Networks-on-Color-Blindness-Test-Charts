# 经典卷积神经网络在色盲测试图表上的对比研究

## 项目简介

本项目系统性地比较了12种经典卷积神经网络（CNN）架构在石原色盲测试图表（Ishihara Color Blindness Test Chart）识别任务上的表现。石原色盲检测图由密集排列的彩色圆点构成，其中数字区域的圆点与背景区域的圆点在色相上相近但可区分——色觉正常者可以辨认数字，而色觉异常者则难以分辨。这一任务对模型的颜色特征提取能力和空间模式识别能力提出了双重挑战。

项目包含完整的实验框架、图形化操作界面以及合成数据集生成器，支持从数据准备、模型训练、性能评估到结果对比的全流程操作。

**作者**：范宇桐、栾华糠

---

## 项目结构

```
.
├── models/                          # CNN模型定义
│   ├── LeNet_5.py                   # LeNet-5
│   ├── AlexNet.py                   # AlexNet
│   ├── VGG.py                       # VGG11 / VGG13 / VGG16 / VGG19
│   ├── ResNet.py                    # ResNet18 / ResNet34 / ResNet50
│   ├── Inception.py                 # InceptionV1(GoogLeNet) / InceptionV3
│   └── DenseNet.py                  # DenseNet121 / DenseNet169
├── experiments/
│   └── proper_grid_search.py        # 网格搜索实验框架（数据加载、训练、评估）
├── data/
│   └── ishihara_dataset/
│       ├── train/                   # 原始训练数据（0-9类别）
│       └── generated/               # 合成数据集（每类6000张，共60000张）
├── cnn_comparison_gui.py            # 图形化操作界面
├── generate_dataset.py              # 合成数据集生成器
├── run_proper_grid_search.py        # 命令行批量实验入口
├── verify_all_models.py             # 模型完整性验证脚本
├── test_gui_fix.py                  # GUI功能测试脚本
└── README.md
```

---

## 支持的模型

本项目实现了12种具有里程碑意义的CNN架构，涵盖了从早期经典网络到现代深度网络的完整发展脉络：

| 系列 | 模型 | 核心创新 | 发表年份 |
|------|------|---------|---------|
| **LeNet** | LeNet-5 | 开创性的卷积+池化结构 | 1998 |
| **AlexNet** | AlexNet | ReLU激活、Dropout正则化、GPU训练 | 2012 |
| **VGG** | VGG11 / VGG13 / VGG16 / VGG19 | 小卷积核(3x3)堆叠替代大卷积核 | 2014 |
| **ResNet** | ResNet18 / ResNet34 / ResNet50 | 残差连接(Shortcut Connection)解决梯度消失 | 2015 |
| **Inception** | InceptionV1(GoogLeNet) / InceptionV3 | 多尺度并行卷积(Inception Module) | 2014/2015 |
| **DenseNet** | DenseNet121 / DenseNet169 | 密集连接(Dense Block)实现特征复用 | 2017 |

---

## 功能特性

### 1. 图形化操作界面（cnn_comparison_gui.py）

基于tkinter构建的完整GUI应用，包含以下功能模块：

- **数据集加载**：支持选择本地数据目录，自动按类别组织，显示各类别样本数量和统计信息
- **模型选择与配置**：从12种CNN架构中选择目标模型，配置学习率、批量大小、训练轮数等超参数
- **模型训练**：后台线程执行训练，实时显示训练进度、损失值和准确率变化
- **训练可视化**：绘制训练/验证损失曲线和准确率曲线，支持保存为图片
- **模型加载与预测**：支持加载已保存的模型权重（兼容多种保存格式），对单张图片进行预测并显示各类别置信度
- **模型对比**：在同一图表中对比多个模型的训练曲线和最终性能

### 2. 合成数据集生成器（generate_dataset.py）

由于真实石原色盲检测图数量有限，本项目开发了合成数据集生成器，能够按需生成大规模训练数据：

**生成算法**：
1. 使用7x5的0/1矩阵定义数字0-9的形状模板
2. 将模板缩放到128x128画布上，确定每个像素属于数字区域还是背景区域
3. 在整个画布上密集铺设随机大小（半径3-7像素）的小圆点，每个圆点带有随机位置偏移
4. 数字区域的圆点使用前景色系（如绿色系），背景区域的圆点使用背景色系（如红色系）
5. 每个圆点内部的每个像素独立施加RGB抖动（±20），模拟真实石原图丰富的颜色变化
6. 对画布进行圆形裁剪，模拟真实色盲检测图的圆形外观

**颜色方案**：内置5种颜色方案（红绿/绿棕/橙绿/黄绿/紫红），每张图随机选择，增加数据多样性。

**安全数据增强**（`safe_augment`函数）：
- 亮度抖动（±10%）：模拟不同光照条件
- 高斯噪声（σ≤3）：模拟图像采集噪声
- 斜向拉伸（0.85~1.15倍）：沿随机角度方向做非等比缩放，让数字形状产生几何变化（变胖/变瘦/变长/变短）
- 轻微缩放+平移（0.92~1.08倍，±5%偏移）：模拟数字位置偏移

**禁止的增强操作**（避免破坏标签语义）：
- 水平翻转：3→反3(像E)，9→反9
- 垂直翻转：9→6，6→9
- 180°旋转：9→6，6→9
- 大角度旋转(>15°)：所有数字变形

### 3. 网格搜索实验框架（experiments/proper_grid_search.py）

系统化的超参数搜索框架，支持：
- 自动遍历模型架构、学习率、批量大小等参数组合
- K折交叉验证评估
- 实验结果自动保存和对比分析

---

## 环境要求

- Python 3.8+
- PyTorch 1.9+
- torchvision
- numpy
- pandas
- matplotlib
- Pillow (PIL)
- tkinter（Python标准库）

---

## 安装依赖

```bash
pip install torch torchvision numpy pandas matplotlib Pillow
```

---

## 使用方法

### 1. 启动图形界面

```bash
python cnn_comparison_gui.py
```

启动后可通过界面完成数据加载、模型训练、预测识别等全部操作。

### 2. 生成合成数据集

```bash
# 默认生成1000张（每类100张）
python generate_dataset.py

# 生成与原数据集等量（每类6000张，共60000张），开启30%增强
python generate_dataset.py -n 6000 --augment --aug-ratio 0.3

# 仅生成指定类别
python generate_dataset.py -d 0 1 2 3 4

# 指定输出目录和随机种子
python generate_dataset.py -o ./my_data -n 500 --seed 42
```

**命令行参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-n` / `--num-per-class` | 每类生成图片数量 | 100 |
| `-d` / `--digits` | 指定生成的数字类别 | 0-9 |
| `-o` / `--output` | 输出目录 | `data/ishihara_dataset/generated` |
| `--img-size` | 图像尺寸 | 128 |
| `--augment` | 开启安全数据增强 | 关闭 |
| `--aug-ratio` | 增强图片占比 | 0.3 |
| `--seed` | 随机种子 | 2024 |

### 3. 运行批量实验

```bash
python run_proper_grid_search.py
```

### 4. 验证模型完整性

```bash
python verify_all_models.py
```

---

## 数据集格式

数据集采用PyTorch `ImageFolder` 兼容的目录结构：

```
data/ishihara_dataset/
├── train/                    # 原始训练数据
│   ├── 0/                    # 数字0
│   │   ├── 0_001.png
│   │   ├── 0_002.png
│   │   └── ...
│   ├── 1/                    # 数字1
│   │   └── ...
│   └── ...                   # 数字2-9
└── generated/                # 合成数据集
    ├── 0/
    │   ├── 0_0.png
    │   ├── 0_1.png
    │   └── ...               # 共6000张
    ├── 1/
    └── ...                   # 数字0-9，每类6000张
```

每张图片为128x128像素的RGB PNG格式，可直接通过 `torchvision.datasets.ImageFolder` 加载：

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder('data/ishihara_dataset/generated', transform=transform)
```

---

## 实验设计

### 研究问题

1. 不同深度的CNN架构在色盲图表识别任务上的表现差异如何？
2. 残差连接（ResNet）、密集连接（DenseNet）、多尺度特征（Inception）等结构创新对该任务是否有效？
3. 合成数据集能否有效替代真实数据集用于模型训练？
4. 数据增强策略对模型泛化能力的影响如何？

### 评估指标

- **准确率（Accuracy）**：模型在测试集上的分类正确率
- **训练效率**：达到指定准确率所需的训练轮数
- **参数量**：模型的可训练参数总数
- **推理速度**：单张图片的平均推理时间

---

## 实验总结

通过本项目，我们：

- 系统性地实现了从LeNet-5到DenseNet的12种经典CNN架构，深入理解了卷积神经网络的发展脉络和核心设计思想
- 构建了完整的实验对比框架，能够公平地评估不同架构在相同数据集和训练条件下的表现
- 开发了合成数据集生成器，通过0/1矩阵定义数字图案、双色系圆点填充、逐像素颜色抖动等策略，成功生成了与真实石原色盲检测图高度相似的合成数据
- 设计了安全的数据增强策略，在增加数据多样性的同时避免了翻转/旋转等会破坏数字语义的操作（如9倒转变成6）
- 掌握了PyTorch模型训练的完整流程，包括数据加载、模型定义、训练循环、学习率调度、模型保存与加载
- 学会了使用tkinter构建功能完整的图形用户界面，实现训练可视化、模型对比、实时预测等交互功能

遇到的主要挑战包括：

- **数据稀缺**：真实色盲检测图数量有限，通过开发合成数据生成器解决了这一问题
- **数据增强语义约束**：数字识别任务中翻转和旋转会改变标签语义（如6和9互为翻转），需要仔细设计增强策略
- **模型格式兼容**：不同版本的模型保存格式存在差异，设计了多层fallback机制确保兼容加载
- **GUI与训练线程协调**：训练过程在后台线程执行，需要正确处理线程间通信和界面更新

通过查阅文献、分析架构和反复调试，我们成功解决了这些问题，完成了一个功能完整、架构清晰的CNN对比研究系统。
