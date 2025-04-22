# coding: utf-8
'''指定文件编码，确保可以文件可以正确处理Unicode字符'''
import os.path
import pickle
import os
import numpy as np
import gzip     #用于读取压缩文件


#定义一个字典，存储MNIST数据集文件的名称
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

#求出当前程序所在的绝对路径上一级，并加上即将要创建的文件名，使得同样可以放在该文件夹下
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

'''训练集，测试集，维度，大小'''
train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

#定义一个函数，加载标签数据
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    return labels
'''
np.frombuffer(): 这是numpy库中的函数，用于从缓冲区（即内存中的一段字节序列）创建一个数组。
    f.read(): 这是传递给np.frombuffer的缓冲区，即上面读取的文件内容。
    np.uint8: 这是指定数组元素的数据类型，uint8代表无符号8位整数，这是MNIST数据集中标签数据的格式。
    offset=8: 这是np.frombuffer的一个参数，表示在创建数组时跳过缓冲区的前8个字节。
    MNIST数据集的标签文件有一个8字节的头部，包含一些元数据，如魔术数字和标签数量，这些信息对于数据的实际内容是不必要的.
'''

#定义一个函数，加载图像数据
def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)   #改为二维数组，第一个参数表示待计算，例如(60000,784)
    print("Done")
    return data

#定义一个函数，存放MNIST的数据集
def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset


def init_mnist():
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")
'''
pickle.dump(dataset, f, -1): 这是pickle模块的dump函数调用，用于将一个Python对象序列化并写入到文件中。
    dataset: 这是要序列化的对象，根据上下文，这通常是一个包含数据集信息的字典。
    f: 这是上面通过open函数打开的文件对象。
    -1: 这是序列化协议的版本号。-1代表使用最高可用的协议版本。
'''


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T
'''
for idx, row in enumerate(T):: 这是一个for循环，遍历数组T中的每一行。
    idx: 这是当前行的索引。
    row: 这是当前行的数据，即T数组中的一行。
    enumerate(T): 这个函数返回一个包含索引和对应元素的元组的迭代器，使得我们可以同时获得行的索引和数据。
'''

#是否正规化，数据展平，one-hot编码
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集

    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label :
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组

    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0


    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


if __name__ == '__main__':
    init_mnist()


