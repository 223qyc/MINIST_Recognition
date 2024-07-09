# 开发日期  2024/6/16
import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    if x.ndim==2:
        x=x.T
        x=x-np.max(x,axis=0)
        y=np.exp(x)/np.sum(np.exp(x),axis=0)
        return y.T
    x=x-np.max(x)
    return np.exp(x)/np.sum(np.exp(x))

def cross_entropy_error(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)

    if t.size==y.size:
        t=t.argmax(axis=1)

    batch_size=y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size

def numerical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x)
    for idx in range(x.size):
        temp_val=x[idx]
        x[idx]=temp_val+h
        fxh1=f(x)
        x[idx]=temp_val-h
        fxh2=f(x)
        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=temp_val

    return grad


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
