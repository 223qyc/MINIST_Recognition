import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    '''
    手写识别-卷积神经网络模型
    '''
    def __init__(self,num_classes=10,dropout=0.2):
        '''
        :param num_classes: 0-9
        :param dropout
        '''
        super(CNNModel, self).__init__()

        # Conv
        self.conv_layers=nn.Sequential(
            # First
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # FC
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        '''
        :param x:  [batch,channel,28,28]
        :return: [batch,num_classes]
        '''
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x



# 测试模型结构
if __name__ == "__main__":
    # 创建模型实例
    model = CNNModel()

    # 打印模型结构
    print(model)

    # 测试前向传播
    batch_size = 64
    x = torch.randn(batch_size, 1, 28, 28)  # 模拟MNIST输入
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

