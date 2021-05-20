import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from fastprogress import master_bar, progress_bar

from data_process.clustering import make_cluster
from model.model import LSTM
from model.losses import RMELoss
import os 



def slidng_windows(data, seq_length):
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

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ctf_indexs, cthh_indexs, cthb_indexs, wif_indexs, wihh_indexs, wihb_indexs = make_cluster(data_path = data_path)
    

    stv_path = os.path.join(data_path, './sales_train_validation.csv')
    stv = pd.read_csv(stv_path).iloc[:, 6:]
    
    ctf_index_1 = ctf_indexs[0]

    X = stv.loc[ctf_index_1].astype(np.int16).T
    print(X.head())

    scaler = StandardScaler()
    scaler = scaler.fit(X)
    X = scaler.transform(X)
    print(X.shape)

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



    ############# Parameters ###################
    num_epochs = 500
    learning_rate = 1e-3
    input_size = np.array(trainX.shape)[2]
    hidden_size = 512
    num_layers   = 1
    num_classes = np.array(trainX.shape)[2]

    ############ Init the Model ################
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
    lstm.to(device)

    ########### Set Criterion Optimizer and scheduler ##############
    #criterion = torch.nn.MSELoss().to(device)
    criterion = RMELoss().to(device)
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5, min_lr=1e-7, eps=1e-08)


    # Train the model
    for epoch in progress_bar(range(num_epochs)):
        lstm.train()
        outputs = lstm(trainX.to(device))
        optimizer.zero_grad()

        # obtain the loss function
        loss = criterion(outputs, trainY.to(device))
        loss.backward()

        optimizer.step()

        #Evaluate on test
        lstm.eval()
        valid = lstm(testX.to(device))
        vall_loss = criterion(valid, testY.to(device))
        scheduler.step(vall_loss)

        if epoch % 50 == 0:
            print("Epoch: %d, loss: %1.5f valid loss: %1.5f " % (epoch, loss.cpu().item(),vall_loss.cpu().item()))


    ############# Prediction ###############
    lstm.eval()
    train_predict = lstm(dataX.to(device))
    data_predict = train_predict.cpu().data.numpy()
    print(data_predict[:5])

    





if __name__ == '__main__':
    main()