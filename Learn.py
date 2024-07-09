# 开发日期  2024/6/16
from TwoLayerNet import TwoLayerNet
from Getdata import  load_mnist
import  numpy as np
import matplotlib.pyplot as plt
# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 梯度
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

## 验证
plt.subplot(1, 2, 1)
plt.plot(np.arange(0, 10000), train_loss_list)
plt.title('Loss')
plt.subplot(1, 2, 2)
plt.plot(np.arange(0, np.size(train_acc_list)), train_acc_list, np.arange(0, np.size(test_acc_list)), test_acc_list)
plt.title('accuracy')
plt.show()