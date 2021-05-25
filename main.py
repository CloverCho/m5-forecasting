import os 
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from fastprogress import master_bar, progress_bar

from data_process.clustering import make_cluster
from model.model import LSTM
from model.losses import RMELoss
from train.LSTM import singleLSTM

def slidng_windows(self, data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)



def main():

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, './data')

    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    ctf_indexs, cthh_indexs, cthb_indexs, wif_indexs, wihh_indexs, wihb_indexs = make_cluster(data_path = data_path)
    

    stv_path = os.path.join(data_path, './sales_train_validation.csv')
    ste_path = os.path.join(data_path, './sales_train_evaluation.csv')
    sub_path = os.path.join(data_path, './sample_submission.csv')

    stv = pd.read_csv(stv_path).iloc[:, 6:]
    ste = pd.read_csv(ste_path).iloc[:, -28:]
    submission = pd.read_csv(sub_path)

    for indexs in [ctf_indexs, cthh_indexs, cthb_indexs, wif_indexs, wihh_indexs, wihb_indexs]:
        for index in indexs:

            X = stv.loc[index].astype(np.int16).T
            print(X.head())

            pred_X = ste.loc[index].astype(np.int16).T
            print(pred_X.head())

            scaler = StandardScaler()
            scaler = scaler.fit(X)
            X = scaler.transform(X)
            print(X.shape)

            pred_scaler = StandardScaler()
            pred_scaler = pred_scaler.fit(pred_X)
            pred_X = pred_scaler.transform(pred_X)
            print(pred_X.shape)
            print('hello')


            seq_length = 28
            x, y = slidng_windows(X, seq_length)
            


            print(x.shape)
            print(y.shape)



            train_size = int(len(y) * 0.67)
            test_size = len(y) - train_size

            dataX = torch.Tensor(np.array(x))
            dataY = torch.Tensor(np.array(y))

            trainX = torch.Tensor(np.array(x[0:train_size]))
            trainY = torch.Tensor(np.array(y[0:train_size]))

            testX = torch.Tensor(np.array(x[train_size:len(x)]))
            testY = torch.Tensor(np.array(y[train_size:len(y)]))



            print("train shape is: ", trainX.size())
            print("train label shape is: ", trainY.size())
            print("test shape is: ", testX.size())
            print("test label shape is: ", testY.size())



            ########### Parameters ###############
            num_epochs = 30
            lr = 1e-3
            ######################################

            model_lstm1 = singleLSTM(trainX=trainX, trainY=trainY, testX=testX, testY=testY, hidden_size=512)
            loss_lstm1, vali_loss_lstm1 = model_lstm1.train(num_epochs=num_epochs, lr=lr)
            print("Epoch: %d,  loss: %1.5f,  validation loss: %1.5f" % (num_epochs, loss_lstm1, vali_loss_lstm1))

            pred_y = model_lstm1.predict(pred_X)

            #print(pred_y[:5])
            submission.loc[index].iloc[:,1:] = pred_y
            submission.iloc[:, 1:] = submission.iloc[:, 1:].astype(np.int16)
    
    
    submission.to_csv(r'./submission.csv', index=False)




if __name__ == '__main__':
    main()
