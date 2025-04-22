# 通用参数
common_params={
    'batch_size': 64,
    'epochs':50,
    'learning_rate':1e-3,
    'weight_decay':1e-5,
    'seed':42,
    'num_classes':10,
    'log_interval':100,
}


# FC
fc_params={
    **common_params,
    'model_name': 'fc',
    'hidden_sizes': [512,256,128],
    'dropout_rate': 0.2,
    'optimizer': 'adam',
    'scheduler': 'step',  # 学习率调度器类型
    'step_size': 5,  # 学习率调度器步长
    'gamma': 0.6,  # 学习率衰减因子
}
# CNN
cnn_params={
    **common_params,
    'model_name': 'cnn',
    'dropout_rate': 0.2,
    'optimizer': 'adam',
    'scheduler': 'step',
    'step_size': 5,
    'gamma': 0.6,
}
# Transformer
transformer_params={
    **common_params,
    'model_name': 'transformer',
    'd_model': 256,  # 模型维度
    'nhead': 8,  # 多头注意力的头数
    'num_layers': 4,  # Transformer编码器层数
    'dim_feedforward': 1024,  # 前馈网络的隐藏层维度
    'dropout_rate': 0.1,
    'optimizer': 'adam',
    'scheduler': 'cosine',  # 余弦退火学习率调度器
    'T_max': 10,  # 余弦周期
}



def get_hyperparams(model_name):
    '''
    :param model_name: 根据对应的模型的名称实现对参数的提取
    :return: 对应模型的超参数字典
    如果名字错误则不提供支持
    '''
    if model_name == 'fc':
        return fc_params
    elif model_name == 'cnn':
        return cnn_params
    elif model_name == 'transformer':
        return transformer_params
    else:
        raise ValueError(f"不支持的模型名称: {model_name}，可选值为'fc', 'cnn', 'transformer'")