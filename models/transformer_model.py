import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class PositionalEncoding(nn.Module):
    '''
    编写位置编码
    '''
    def __init__(self,d_model,max_len=1000):
        '''
        :param d_model:词向量维度
        :param max_len: 序列最大长度
        '''
        super(PositionalEncoding, self).__init__()

        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用sin
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self,x):
        '''
        :return: 带有位置编码的嵌入
        '''
        return x + self.pe[:, :x.size(1),:]

class TransformerModel(nn.Module):
    '''
    定义Transformer-MINIST
    '''
    def __init__(self,input_dim=28,dmodel=256,head=8,num_layers=4,dim_ffn=1024,num_classes=10,dropout=0.1):
        '''
        :param input_dim: 输入维度，看成28个长度为28的seq
        :param dmodel:嵌入维度
        :param head:头数
        :param num_layers:层数
        :param dim_ffn:FFN的维度
        :param num_classes:0-9
        :param dropout
        '''
        super(TransformerModel, self).__init__()

        # 嵌入与编码处理
        self.embedding = nn.Linear(input_dim,dmodel)
        self.positional_encoding = PositionalEncoding(dmodel)

        # 编码器层
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=dmodel,
            nhead=head,
            dim_feedforward=dim_ffn,
            dropout=dropout,
            batch_first=True
        )

        # 编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,num_layers=num_layers)
        # fc
        self.fc = nn.Linear(dmodel,num_classes)
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        '''
        :param x: [batch_size, 1, 28, 28]
        :return: [batch_size, num_classes]
        '''

        #pipeline
        x=x.squeeze(1)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x=self.transformer_encoder(x)
        x=torch.mean(x, dim=1)
        x=self.dropout(x)
        x = self.fc(x)

        return x


# 测试模型结构
if __name__ == "__main__":
    model = TransformerModel()
    print(model)

    batch_size = 64
    x = torch.randn(batch_size, 1, 28, 28)  # 模拟输入
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")