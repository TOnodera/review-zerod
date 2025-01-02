
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

# 交差エントロピー誤差
def cross_entropy_error(y, t):
    # このdeltaを足さないとy=0の時に-infになる
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))