'''
#1
import numpy as np
import matplotlib.pyplot as plt

N = 120
x = np.random.rand(N)
y = np.random.rand(N)

colors = np.random.rand(N)
size = np.array(
    [321 ,717 ,950 ,264 ,667 ,692 ,509 ,39 ,234 ,
1192 ,512 ,10 ,1664 ,79 ,252 ,168 ,759 ,29 ,
27 ,974 ,914 ,657 ,1115 ,347 ,963 ,892 ,10 ,
1 ,405 ,211 ,258 ,48 ,3 ,240 ,123 ,23 ,814 ,
42 ,188 ,75 ,637 ,298 ,65 ,125 ,53 ,41 ,6 ,
407 ,393 ,211 ,95 ,87 ,447 ,86 ,8 ,705 ,8 ,1 ,
10 ,272 ,182 ,537 ,203 ,183 ,64 ,38 ,199 ,42 ,
467 ,306 ,62 ,285 ,40 ,18 ,306 ,139 ,237 ,672 ,
329 ,2 ,155 ,583 ,181 ,209 ,19 ,37 ,96 ,277 ,
63 ,209 ,515 ,25 ,5 ,3 ,47 ,612 ,43 ,20 ,28 ,
14 ,59 ,57 ,94 ,84 ,15 ,10 ,115 ,137 ,1 ,20 ,
59 ,283 ,1 ,40 ,11 ,26 ,183 ,172 ,317 ,11])
area = np.pi * (size/50)**2
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()
'''
#2.
import os 
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from data_process.clustering import make_cluster

def main():

    ########### Parameters ###############
    num_epochs = 200
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
    # lgbm_ste = pd.read_csv(ste_path).iloc[:, -lgbm_period:]
    submission = pd.read_csv(sub_path)



    target = ctf_indexs[0]
    
    targetdf = stv.loc[target].astype(np.int16)

    print(targetdf)

    date_cols = [c for c in stv.columns if 'd_' in c]
    aggr_array = []
    targetdf = targetdf[date_cols]
    print(targetdf)
    
    for d in date_cols:
        aggr_array.append(targetdf[d].values.sum())

    daily_time_series_df = pd.DataFrame(data=aggr_array, columns=['Sales'], index=date_cols)
    series = daily_time_series_df['Sales']
    print(series)   


    X_values = range(len(targetdf.columns))
    coeffs = np.polyfit(X_values, series.values, 7)
    poly_eqn = np.poly1d(coeffs)
    poly_y = poly_eqn(X_values)

    
    plt.plot(date_cols, poly_y, label='ctf cluster 0', linewidth=2, color = 'green')
    plt.plot(date_cols, series.values, linewidth=2, color='green', alpha=0.2)
    
    plt.legend()
    plt.show()
