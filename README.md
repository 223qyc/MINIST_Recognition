# 写在前面
该仓库汇总记录了两个关于手写识别的内容：
- **第一个**：无框架手动实现，只有简单的指标测评和图像绘制，使用最简单的MLP实现，记录时间于2024年6月。是第一次接触到深度学习后第一次实现，有很大纪念意义。
- **第二个**：更工业化的代码体系，更健壮的项目结构，记录于2025年4月。
# MNIST手写数字识别系统

## 项目概述

本项目实现了基于PyTorch的MNIST手写数字识别系统，提供了三种不同的深度学习模型架构：全连接神经网络(FC)、卷积神经网络(CNN)和Transformer模型。系统支持模型训练、测试和结果可视化，并通过TensorBoard提供训练过程监控。

## 环境配置

### 依赖库

```bash
python
torch
torchvision
matplotlib
seaborn
scikit-learn
tensorboard
tqdm
numpy
```

### 安装方法

```bash
# 建议使用conda创建虚拟环境
conda create -n mnist python=3.8
conda activate mnist

# 安装依赖库
pip install torch torchvision matplotlib seaborn scikit-learn tensorboard tqdm numpy
```

## 项目结构

```
├── config/                  # 配置文件目录
│   └── hyperparams.py       # 模型超参数配置
├── data/                    # 数据目录
│   └── MNIST/               # MNIST数据集
├── models/                  # 模型定义
│   ├── fc_model.py          # 全连接网络模型
│   ├── cnn_model.py         # 卷积神经网络模型
│   └── transformer_model.py # Transformer模型
├── runs/                    # 训练结果和日志
├── train.py                 # 训练脚本
└── test.py                  # 测试和评估脚本
```

## 模型架构

### 全连接网络 (FC)

全连接网络模型采用多层感知机结构，将28×28的图像展平为784维向量输入，通过多个全连接层和ReLU激活函数进行特征提取和分类。

主要特点：
- 输入层：784个神经元（28×28像素）
- 隐藏层：可配置的多层结构，默认为[512, 256, 128]
- 输出层：10个神经元（对应0-9数字）
- Dropout正则化：防止过拟合

### 卷积神经网络 (CNN)

卷积神经网络模型利用卷积层提取图像的空间特征，通过池化层降维，最后使用全连接层进行分类。

主要特点：
- 三层卷积结构：32→64→128通道
- 批归一化：提高训练稳定性
- 最大池化：特征降维
- 全连接分类器：128×3×3 → 256 → 10

### Transformer模型

Transformer模型将图像视为序列数据处理，利用自注意力机制捕捉像素间的长距离依赖关系。

主要特点：
- 将28×28图像视为28个长度为28的序列
- 位置编码：提供序列位置信息
- 多头自注意力机制：捕捉像素间关系
- 可配置的编码器层数和维度

## 使用方法

### 训练模型

```bash
# 训练全连接网络
python train.py --model_type fc --num_workers 4

# 训练CNN模型
python train.py --model_type cnn --num_workers 4

# 训练Transformer模型
python train.py --model_type transformer --num_workers 4
```

训练参数说明：
- `--model_type`: 选择模型类型，可选值为fc、cnn、transformer
- `--num_workers`: 数据加载线程数

### 测试模型

```bash
# 测试已训练的模型
python test.py --model_type fc --model_path runs/fc_YYYYMMDD_HHMMSS/fc_best.pth
```

测试参数说明：
- `--model_type`: 选择模型类型，可选值为fc、cnn、transformer
- `--model_path`: 模型权重文件路径

## 结果可视化

系统提供多种可视化方式：

1. **训练过程监控**：
   - 使用TensorBoard查看训练和验证的损失曲线、准确率曲线
   - 启动方式：`tensorboard --logdir=runs`

2. **测试结果可视化**：
   - 混淆矩阵：展示分类结果的详细分布
   - 精确率-召回率曲线：评估模型在各类别上的性能
   - ROC曲线：评估模型的分类性能
   - 错误预测样本可视化：直观展示模型的错误预测

所有可视化结果保存在对应模型的`runs/[model_type]_[timestamp]/results/`目录下。

## 超参数配置

可以在`config/hyperparams.py`文件中修改各模型的超参数：

- 通用参数：批量大小、学习率、权重衰减等
- FC模型参数：隐藏层大小、Dropout率等
- CNN模型参数：Dropout率、优化器设置等
- Transformer模型参数：模型维度、注意力头数、编码器层数等

## 性能比较

三种模型在MNIST数据集上的典型性能：

| 模型 | 准确率 | 训练时间 | 参数量 |
|-----|-------|---------|-------|
| FC  | ~98%  | 快      | 少    |
| CNN | ~99%  | 中      | 中    |
| Transformer | ~99% | 慢 | 多 |

不同模型适用于不同场景：FC模型简单高效，CNN模型准确度高，Transformer模型具有更强的特征提取能力但计算开销较大。