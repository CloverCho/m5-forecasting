import torch
import numpy as np
import pandas as pd

from model.model import Encoder
# from model.model import Decoder
from model.model import AttentionDecoder
from model.model import Attention
from model.model import Seq2Seq

from model.losses import RMELoss
from fastprogress import master_bar, progress_bar


class EDA_Train():
    def __init__(self, X, train_ratio=0.67, hidden_size=512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = torch.device('cpu')

        self.seq_length = 28
        self.labels_length = 28


        x, y = self.sliding_windows(X, self.seq_length, self.labels_length)
        print(len(x))
        print(len(y))

        train_size = int(len(y) * train_ratio)
        test_size = len(y) - train_size

        self.trainX = torch.Tensor(np.array(x[0:train_size]))
        self.trainY = torch.Tensor(np.array(y[0:train_size]))
        self.testX = torch.Tensor(np.array(x[train_size:len(x)]))
        self.testY = torch.Tensor(np.array(y[train_size:len(y)]))
        print(self.trainX.shape)
        print(self.trainY.shape)
        print(self.testX.shape)
        print(self.testY.shape)

        self.num_layers = 1
        self.hidden_size = hidden_size
        self.input_size = np.array(self.trainX.shape)[2]
        self.num_classes = np.array(self.trainX.shape)[2]


        #n_features = 1 --> num_classes

        self.model = Seq2Seq(self.seq_length, self.num_classes, self.input_size, self.hidden_size)

    def sliding_windows(self, data, seq_length,labels_length):

        x = []
        y = []

        for i in range(len(data)-(seq_length+labels_length)):
            _x = data[i:(i+seq_length)]
            _y = data[(i+seq_length):(i+seq_length+labels_length)]
            x.append(_x)
            y.append(_y)

        return np.array(x), np.array(y)

    def train(self, num_epochs=30, lr=1e-3):

        criterion = RMELoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5, min_lr=1e-7,
                                                               eps=1e-08)
        mb = master_bar(range(1, num_epochs + 1))
        # Train the model
        for epoch in mb:
            self.model.train()

            train_losses = []

            for i in progress_bar(range(self.trainX.size()[0]), parent=mb):
                optimizer.zero_grad()

                seq_inp = self.trainX[i,:,:].to(self.device)
                seq_true = self.trainY[i,:,:].to(self.device)
                outputs = self.model(seq_inp, seq_inp[self.seq_length-1:self.seq_length,:])

                # obtain the loss function
                loss = criterion(outputs, seq_true)
                train_losses.append(loss.item())

            val_losses = []
            for i in progress_bar(range(self.testX.size()[0]), parent = mb):
                seq_inp = self.testX[i, :, :].to(self.device)
                seq_true = self.testY[i, :, :].to(self.device)
                outputs = self.model(seq_inp, seq_inp[self.seq_length - 1:self.seq_length, :])

                # obtain the loss function
                loss = criterion(outputs, seq_true)
                val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)

        return train_loss.item(), val_loss.item()

    def predict(self, pred_X):

        self.model.eval()
        pred_data = torch.Tensor(np.expand_dims(np.array(pred_X), axis=0)).to(self.device)
        # print('model input')
        # print(pred_data.size())

        # print('prev_y input')
        prev_y = pred_data[0, self.seq_length-1:self.seq_length,:]
        # print(prev_y.size())

        pred_d1914 = self.model(pred_data, prev_y).cpu().data.numpy()
        pred_y = np.copy(pred_d1914)

        
        # print(pred_d1914)

        for i in range(1, 28):
            pred_data = np.array(pred_X[i:])
            print('---------------')
            # print(pred_data.shape)
            # print(pred_y.shape)

            pred_data = np.concatenate((pred_data, pred_y), axis=0)
            pred_data = torch.Tensor(np.expand_dims(pred_data, axis=0)).to(self.device)
            pred_result = self.model(pred_data,pred_data[0, self.seq_length-1:self.seq_length,:]).cpu().data.numpy()
            pred_y = np.concatenate((pred_y, pred_result), axis=0)

        pred_y = pred_y.T

        return pred_y
