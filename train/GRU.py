import torch
import numpy as np
import pandas as pd

from model.model import GRU
from model.losses import RMELoss
from fastprogress import master_bar, progress_bar


class GRU_Train():
    def __init__(self, X, train_ratio=0.67, hidden_size=512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')

        seq_length = 28
        x, y = self.slidng_windows(X, seq_length)

        train_size = int(len(y) * train_ratio)
        test_size = len(y) - train_size

        self.trainX = torch.Tensor(np.array(x[0:train_size]))
        self.trainY = torch.Tensor(np.array(y[0:train_size]))
        self.testX = torch.Tensor(np.array(x[train_size:len(x)]))
        self.testY = torch.Tensor(np.array(y[train_size:len(y)]))

        self.num_layers = 2
        self.hidden_size = hidden_size
        self.input_size = np.array(self.trainX.shape)[2]
        self.num_classes = np.array(self.trainX.shape)[2]

        self.model = GRU(self.num_classes, self.input_size, self.hidden_size, self.num_layers).to(self.device)

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

        criterion = RMELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5, min_lr=1e-7,
                                                               eps=1e-08)

        print("GRU Train")

        # Train the model
        for epoch in progress_bar(range(num_epochs)):
            self.model.train()
            outputs = self.model(self.trainX.to(self.device))
            optimizer.zero_grad()

            # obtain the loss function
            loss = criterion(outputs, self.trainY.to(self.device))
            valid = self.model(self.testX.to(self.device))
            vali_loss = criterion(valid, self.testY.to(self.device))
            scheduler.step(vali_loss)

            loss_value = loss.cpu().item()
            vali_value = vali_loss.cpu().item()
            
        return loss_value, vali_value

    def predict(self, pred_X):

        self.model.eval()
        pred_data = torch.Tensor(np.expand_dims(np.array(pred_X), axis=0))

        pred_d1914 = self.model(pred_data.to(self.device)).cpu().data.numpy()
        pred_y = np.copy(pred_d1914)

        for i in range(1, 28):
            pred_data = np.array(pred_X[i:])
            pred_data = np.concatenate((pred_data, pred_y), axis=0)
            pred_data = torch.Tensor(np.expand_dims(pred_data, axis=0))
            pred_result = self.model(pred_data.to(self.device)).cpu().data.numpy()
            pred_y = np.concatenate((pred_y, pred_result), axis=0)

        pred_y = pred_y.T

        return pred_y
