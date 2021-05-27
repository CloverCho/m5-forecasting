import torch
import numpy as np

from model.model import SingleRNN
from model.losses import RMELoss
from sklearn.preprocessing import StandardScaler
from fastprogress import master_bar, progress_bar


class singleRNN():
    def __init__(self, trainX, trainY, testX, testY, hidden_size=512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_layers = 1

        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

        self.hidden_size = hidden_size
        self.input_size = np.array(self.trainX.shape)[2]
        self.num_classes = np.array(self.trainX.shape)[2]

        self.model = SingleRNN(self.num_classes, self.input_size, self.hidden_size, self.num_layers).to(self.device)

    def slidng_windows(self, data, seq_length):
        x = []
        y = []

        for i in range(len(data) - seq_length - 1):
            _x = data[i:(i + seq_length)]
            _y = data[i + seq_length]
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y)

    def train(self, num_epochs=30, lr=1e-3):
        criterion = RMELoss.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5, min_lr=1e-7,
                                                               eps=1e-08)

        # Train the model

