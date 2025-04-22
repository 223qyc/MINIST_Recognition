import torch
import torch.nn as nn
import torch.nn.functional as F


class FCModel(nn.Module):
    '''
    手写识别-全连接网络模型
    '''
    def __init__(self,input_size=784,hidden_size=[512,256,128],num_classes=10,dropout=0.2):
        '''
        :param input_size: 28*28
        :param hidden_size: 中间层神经元列表
        :param num_classes: 0-9
        :param dropout
        '''
        super(FCModel, self).__init__()
        layers = []

        layers.append(nn.Linear(input_size, hidden_size[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for i in range(len(hidden_size)-1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_size[-1], num_classes))

        self.layers=nn.Sequential(*layers)


    def forward(self,x):
        '''

        :param x: 输入[batch,channel,28,28]->灰度=1
        :return: 输出[batch,num_classes]
        '''

        x=x.view(x.size(0),-1)
        x = self.layers(x)

        return x


# 内部测试模型结构
if __name__=='__main__':
    model = FCModel()
    print(model)

    batch_size = 64
    x = torch.randn(batch_size, 1, 28, 28)  # 模拟MNIST输入
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")