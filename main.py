import os 
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gc
import time

from sklearn.preprocessing import StandardScaler

from data_process.clustering import make_cluster
from train.SingleLSTM import singleLSTM_Train
from train.SingleGRU import singleGRU_Train
from train.RNN import singleRNN_Train
from train.LSTM import LSTM_Train
from train.GRU import GRU_Train
from train.LGBM import LGBM_Train
from train.EDA import EDA_Train

def main():

    ########### Parameters ###############
    num_epochs = 10
    lr = 1e-3
    lgbm_period = 30
    ######################################



    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, './data')

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    ctf_indexs, cthh_indexs, cthb_indexs, wif_indexs, wihh_indexs, wihb_indexs = make_cluster(data_path = data_path, n_clusters = 20)
    

    stv_path = os.path.join(data_path, './sales_train_validation.csv')
    ste_path = os.path.join(data_path, './sales_train_evaluation.csv')
    sub_path = os.path.join(data_path, './sample_submission.csv')

    stv = pd.read_csv(stv_path).iloc[:, 6:]
    ste = pd.read_csv(ste_path).iloc[:, -28:]
    lgbm_ste = pd.read_csv(ste_path).iloc[:, -lgbm_period:]
    submission = pd.read_csv(sub_path)




    start_time = time.time()


    for indexs in [ctf_indexs, cthh_indexs, cthb_indexs, wif_indexs, wihh_indexs, wihb_indexs]:
        for index in indexs:
            

            X = stv.loc[index].astype(np.int16).T
            #print(X.head())

            pred_X = ste.loc[index].astype(np.int16).T
            #print(pred_X.head())


            lgbm_X = np.array(stv.loc[index].astype(np.int16).T)
            lgbm_pred_X = np.array(lgbm_ste.loc[index].astype(np.int16))

            scaler = StandardScaler()
            scaler = scaler.fit(X)
            X = scaler.transform(X)
            #print(X.shape)

            

            
            pred_scaler = StandardScaler()
            pred_scaler = pred_scaler.fit(pred_X)
            pred_X = pred_scaler.transform(pred_X)
            #pred_X = scaler.transform(pred_X)
            print(pred_X.shape)


            #lgbm_scaler = StandardScaler()
            #lgbm_scaler = lgbm_scaler.fit(lgbm_pred_X)
            #lgbm_pred_X = lgbm_scaler.transform(lgbm_pred_X)



            train_ratio = 0.67
            hidden_size = 512

            
            
            model_lstm1 = singleLSTM_Train(X, train_ratio=train_ratio, hidden_size=hidden_size)
            model_lstm2 = LSTM_Train(X, train_ratio=train_ratio, hidden_size=hidden_size)
            model_rnn = singleRNN_Train(X, train_ratio=train_ratio, hidden_size=hidden_size)
            model_gru1 = singleGRU_Train(X, train_ratio=train_ratio, hidden_size=hidden_size)
            model_gru2 = GRU_Train(X, train_ratio=train_ratio, hidden_size=hidden_size)
            model_eda = EDA_Train(X, train_ratio=train_ratio, hidden_size=hidden_size)

            model_opt = None
            loss_opt = 987654321
            vali_loss_opt = 987654321

            #models = [model_lstm1, model_lstm2, model_gru1, model_gru2, model_rnn, model_eda]
            #models = [model_lstm1, model_lstm2, model_gru1, model_gru2, model_rnn]
            models = [model_eda]

            for idx, model in enumerate(models):
                loss, vali_loss = model.train(num_epochs=num_epochs, lr=lr)
                
                if vali_loss < vali_loss_opt:
                    loss_opt = loss
                    vali_loss_opt = vali_loss
                    model_opt = model
                

            
            
            ############# Prediction ###############
              
            pred_y = model_opt.predict(pred_X)                
            pred_y = pred_scaler.inverse_transform(pred_y.T).T


            
            pred_y[pred_y < 0] = 0
            pred_y = np.rint(pred_y)

            print(pred_y[:5])

            submission.iloc[index,1:] = np.copy(pred_y)

            


            '''
            ########## LGBM ###################
            print(lgbm_X.shape)
            model_lgbm = LGBM_Train(lgbm_X, period=lgbm_period)
            model_lgbm.train()
            pred_y = model_lgbm.predict(lgbm_pred_X)
            #pred_y = lgbm_scaler.inverse_transform(pred_y.T)


            print(pred_y[:5])
            
            submission.iloc[index, 1:] = np.copy(pred_y)
            '''
            
    end_time = time.time()
    print("Training time: {} seconds".format(end_time - start_time))

    submission.to_csv(r'./submission_CHOandSON_EDA.csv', index=False)




if __name__ == '__main__':
    main()
