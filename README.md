# 写在前面
本仓库共记录了两个项目：
- 1.第一次学习深度学习手动实现的手写识别，无框架实现了多层感知机，较为粗糙
  - 完成时间：2024年6月
- 2.第二次记录多种方法的对比，并实现了更强的系统性，也是自己学习的一个跨越。还包括一个[模型比较分析文档](model_comparison.md)
  - 完成时间：2025年4月

以下的项目描述集中于第二次。

# MNIST手写数字识别系统

## 项目概述

本项目实现了基于PyTorch的MNIST手写数字识别系统，提供了三种不同的深度学习模型架构：全连接神经网络(FC)、卷积神经网络(CNN)和Transformer模型。系统具备完整的训练、测试和结果可视化功能，支持TensorBoard监控训练过程。

## 模型架构

### 全连接网络 (FC)

全连接网络模型采用多层感知机结构，将28×28的图像展平为784维向量输入，通过多个全连接层和ReLU激活函数进行特征提取和分类。

### 卷积神经网络 (CNN)

卷积神经网络模型利用卷积层提取图像的空间特征，通过池化层降维，最后使用全连接层进行分类。模型采用三层卷积结构，配合批归一化和最大池化实现特征提取。

### Transformer模型

Transformer模型将图像视为序列数据处理，利用自注意力机制捕捉像素间的长距离依赖关系。模型包含位置编码、多头自注意力机制和前馈神经网络结构。

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

### 测试模型

```bash
# 测试已训练的模型
python test.py --model_type fc --model_path runs/fc_YYYYMMDD_HHMMSS/fc_best.pth
```

## 结果可视化

系统提供多种可视化方式：

1. **训练过程监控**：使用TensorBoard查看训练和验证的损失曲线、准确率曲线
2. **测试结果可视化**：混淆矩阵、精确率-召回率曲线、ROC曲线和错误预测样本可视化

详细的模型性能比较和分析请参考[模型比较分析](model_comparison.md)。
