import torch
import numpy as np


from model.model import LSTM
from model.losses import RMELoss
from sklearn.preprocessing import StandardScaler




class singleLSTM():
    def __init__(X, pred_X):
        self.X = X
        self.pred_X = pred_X
        



    def slidng_windows(data, seq_length):
        x = []
        y = []

        for i in range(len(data) - seq_length - 1):
            _x = data[i:(i+seq_length)]
            _y = data[i+seq_length]
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y)

