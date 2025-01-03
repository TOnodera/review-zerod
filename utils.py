
import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1/(1+np.exp(x))

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c) # オーバーフロー対策
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

# 2乗和誤差
def mean_squered_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 交差エントロピー誤差(バッチ処理対応)
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    # 一括処理して平均を出す
    return -np.sum(t*np.log(y)) / batch_size