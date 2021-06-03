import torch
import numpy as np
import pandas as pd
import lightgbm as lgb

from model.losses import RMELoss
from fastprogress import master_bar, progress_bar



class LGBM_Train:
    def __init__(self, X, period=30):

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.data = X
        self.period = period

        self.numDate = X.shape[0]
        self.numData = X.shape[1]

        #print(self.data.shape)

    def input_label_split(self):
        
        inputs = []
        labels = []

        for data in range(self.numData):

            for date in range(self.numDate - self.period):
                
                _x = self.data[ date : date + self.period, data]
                _y = self.data[ date + self.period, data]
            
                inputs.append(_x)
                labels.append(_y)

        
        inputs = np.array(inputs)
        labels = np.array(labels)

        #print ("Shape of input: {}\nShape of label: {}\n".format(inputs.shape, labels.shape))

        return inputs, labels


    def train(self,  params = None, num_boost_round=2500, early_stopping_rounds=50, verbose_eval=100):


        if params is not None:
            self.params = parmas
        else:
            self.params = {
                    'boosting_type': 'gbdt',
                    'metric': 'rmse',
                    'objective': 'regression',
                    'n_jobs': -1,
                    'seed': 236,
                    'learning_rate': 0.1,
                    'bagging_fraction': 0.75,
                    'bagging_freq': 10, 
                    'colsample_bytree': 0.75}


        trainX, trainY = self.input_label_split() 
        train_set = lgb.Dataset(trainX, trainY)

        self.model = lgb.train(self.params, train_set, num_boost_round = num_boost_round, early_stopping_rounds=early_stopping_rounds, valid_sets = [train_set], verbose_eval=verbose_eval)



    def predict(self, pred_X):
        
        pred_data = pred_X
        
        for i in range(28):
            
            pred_y = self.model.predict(pred_data)
            pred_data = np.concatenate((pred_data[:, 1:], pred_y), axis=1)

        pred_result =  pred_data[:, -28:].astype(int)
        pred_result[pred_result < 0] = 0

        return pred_result
        