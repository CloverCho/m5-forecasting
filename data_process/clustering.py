import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans


def make_cluster(data_path = '../data'):



    CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
            "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
            "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }
    PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }
    SALES_DTYPES = {'id':"category", 'item_id':"category", "dept_id":"category", "cat_id":"category", "state_id":"category"}

    wdays = ['Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    events = ['MemorialDay', 'NBAFinalsStart', 'Ramadan starts', 'NBAFinalsEnd', 'Father\'s Day']


    cal_path = os.path.join(data_path, './calendar.csv')
    ss_path = os.path.join(data_path, './sell_prices.csv')
    stv_path = os.path.join(data_path, './sales_train_validation.csv')

    cal = pd.read_csv(cal_path, dtype=CAL_DTYPES)
    ss = pd.read_csv(ss_path, dtype=PRICE_DTYPES)
    stv = pd.read_csv(stv_path, dtype=SALES_DTYPES)
    print(cal.head())
    print(stv.head())

    print(stv['state_id'].unique())

    group_ct = stv[stv['state_id'] != 'WI']
    group_wi = stv[stv['state_id'] == "WI"]


    print(group_ct.head())
    print(group_wi.head())


    print(group_ct['cat_id'].unique())




    group_ctf = group_ct[group_ct['cat_id'] == 'FOODS']
    group_cthh = group_ct[group_ct['cat_id'] == 'HOUSEHOLD']
    group_cthb = group_ct[group_ct['cat_id'] == 'HOBBIES']

    group_wif = group_wi[group_wi['cat_id'] == 'FOODS']
    group_wihh = group_wi[group_wi['cat_id'] == 'HOUSEHOLD']
    group_wihb = group_wi[group_wi['cat_id'] == 'HOBBIES']


    group_ctf = group_ctf.iloc[:, 6:]
    group_ctf.columns = list(map(lambda x: dict(zip(cal['d'], cal['date']))[x], group_ctf.columns))
    for col in group_ctf.columns:
        group_ctf[col] = group_ctf[col].astype(np.int16)


    group_cthh = group_cthh.iloc[:, 6:]
    group_cthh.columns = list(map(lambda x: dict(zip(cal['d'], cal['date']))[x], group_cthh.columns))
    for col in group_cthh.columns:
        group_cthh[col] = group_cthh[col].astype(np.int16)


    group_cthb = group_cthb.iloc[:, 6:]
    group_cthb.columns = list(map(lambda x: dict(zip(cal['d'], cal['date']))[x], group_cthb.columns))
    for col in group_cthb.columns:
        group_cthb[col] = group_cthb[col].astype(np.int16)


    group_wif = group_wif.iloc[:, 6:]
    group_wif.columns = list(map(lambda x: dict(zip(cal['d'], cal['date']))[x], group_wif.columns))
    for col in group_wif.columns:
        group_wif[col] = group_wif[col].astype(np.int16)

    group_wihh = group_wihh.iloc[:, 6:]
    group_wihh.columns = list(map(lambda x: dict(zip(cal['d'], cal['date']))[x], group_wihh.columns))
    for col in group_wihh.columns:
        group_wihh[col] = group_wihh[col].astype(np.int16)

    group_wihb = group_wihb.iloc[:, 6:]
    group_wihb.columns = list(map(lambda x: dict(zip(cal['d'], cal['date']))[x], group_wihb.columns))
    for col in group_wihb.columns:
        group_wihb[col] = group_wihb[col].astype(np.int16)

    print(group_wif.head())

    #################
    #  Ctf cluster  #
    #################

    ctf_cluster_temp = group_ctf.T.merge(cal.set_index('date'), left_index=True, right_index=True, validate='1:1')
    print(ctf_cluster_temp.head())
    print(ctf_cluster_temp.columns[:-13])

    ctf_cluster_wday = ctf_cluster_temp.groupby('wday').mean()[ctf_cluster_temp.columns[:-13]].T
    ctf_cluster_month = ctf_cluster_temp.groupby('month').mean()[ctf_cluster_temp.columns[:-13]].T
    ctf_cluster_event = ctf_cluster_temp.groupby('event_name_1').mean()[ctf_cluster_temp.columns[:-13]].T[events]
    print(ctf_cluster_wday.head())
    print(ctf_cluster_month.head())
    print(ctf_cluster_event.head())

    # normalize
    ctf_cluster_event = ctf_cluster_event.div(ctf_cluster_wday.sum(axis=1), axis=0)
    ctf_cluster_wday = ctf_cluster_wday.div(ctf_cluster_wday.sum(axis=1), axis=0)
    ctf_cluster_month = ctf_cluster_month.div(ctf_cluster_month.sum(axis=1), axis=0)
    print(ctf_cluster_wday.head())
    print(ctf_cluster_month.head())
    print(ctf_cluster_event.head())


    ctf_cluster_wday = ctf_cluster_wday.set_axis(wdays, axis='columns')
    ctf_cluster_month = ctf_cluster_month.set_axis(months, axis='columns')
    print(ctf_cluster_wday.head())
    print(ctf_cluster_month.head())

    ctf_cluster = ctf_cluster_wday.merge(ctf_cluster_month, left_index=True, right_index=True, validate='1:1')
    ctf_cluster = ctf_cluster.merge(ctf_cluster_event, left_index=True, right_index=True, validate='1:1')
    print(ctf_cluster.head())


    # Clustering
    n_clusters = 5
    km_ctf = KMeans(n_clusters = n_clusters, random_state=0).fit(ctf_cluster)
    ctf_cluster_label = km_ctf.labels_
    print(ctf_cluster_label)

    ctf_cluster_1 = ctf_cluster.iloc[ctf_cluster_label == 0, :]
    ctf_cluster_2 = ctf_cluster.iloc[ctf_cluster_label == 1, :]
    ctf_cluster_3 = ctf_cluster.iloc[ctf_cluster_label == 2, :]
    ctf_cluster_4 = ctf_cluster.iloc[ctf_cluster_label == 3, :]
    ctf_cluster_5 = ctf_cluster.iloc[ctf_cluster_label == 4, :]

    print(ctf_cluster_1.head())



    #################
    #  Cthh cluster #
    #################


    cthh_cluster_temp = group_cthh.T.merge(cal.set_index('date'), left_index=True, right_index=True, validate='1:1')
    cthh_cluster_wday = cthh_cluster_temp.groupby('wday').mean()[cthh_cluster_temp.columns[:-13]].T
    cthh_cluster_month = cthh_cluster_temp.groupby('month').mean()[cthh_cluster_temp.columns[:-13]].T
    cthh_cluster_event = cthh_cluster_temp.groupby('event_name_1').mean()[cthh_cluster_temp.columns[:-13]].T[events]

    # normalize
    cthh_cluster_event = cthh_cluster_event.div(cthh_cluster_wday.sum(axis=1), axis=0)
    cthh_cluster_wday = cthh_cluster_wday.div(cthh_cluster_wday.sum(axis=1), axis=0)
    cthh_cluster_month = cthh_cluster_month.div(cthh_cluster_month.sum(axis=1), axis=0)


    # Rename columns
    cthh_cluster_wday = cthh_cluster_wday.set_axis(wdays, axis='columns')
    cthh_cluster_month = cthh_cluster_month.set_axis(months, axis='columns')

    # Merge
    cthh_cluster = cthh_cluster_wday.merge(cthh_cluster_month, left_index=True, right_index=True, validate='1:1')
    cthh_cluster = cthh_cluster.merge(cthh_cluster_event, left_index=True, right_index=True, validate='1:1')

    # Clustering
    n_clusters = 5
    km_cthh = KMeans(n_clusters = n_clusters, random_state=1).fit(cthh_cluster)
    cthh_cluster_label = km_cthh.labels_

    cthh_cluster_1 = cthh_cluster.iloc[cthh_cluster_label == 0, :]
    cthh_cluster_2 = cthh_cluster.iloc[cthh_cluster_label == 1, :]
    cthh_cluster_3 = cthh_cluster.iloc[cthh_cluster_label == 2, :]
    cthh_cluster_4 = cthh_cluster.iloc[cthh_cluster_label == 3, :]
    cthh_cluster_5 = cthh_cluster.iloc[cthh_cluster_label == 4, :]




    #################
    #  Cthb cluster #
    #################


    cthb_cluster_temp = group_cthb.T.merge(cal.set_index('date'), left_index=True, right_index=True, validate='1:1')
    cthb_cluster_wday = cthb_cluster_temp.groupby('wday').mean()[cthb_cluster_temp.columns[:-13]].T
    cthb_cluster_month = cthb_cluster_temp.groupby('month').mean()[cthb_cluster_temp.columns[:-13]].T
    cthb_cluster_event = cthb_cluster_temp.groupby('event_name_1').mean()[cthb_cluster_temp.columns[:-13]].T[events]

    # normalize
    cthb_cluster_event = cthb_cluster_event.div(cthb_cluster_wday.sum(axis=1), axis=0)
    cthb_cluster_wday = cthb_cluster_wday.div(cthb_cluster_wday.sum(axis=1), axis=0)
    cthb_cluster_month = cthb_cluster_month.div(cthb_cluster_month.sum(axis=1), axis=0)

    # Rename columns
    cthb_cluster_wday = cthb_cluster_wday.set_axis(wdays, axis='columns')
    cthb_cluster_month = cthb_cluster_month.set_axis(months, axis='columns')

    # Merge
    cthb_cluster = cthb_cluster_wday.merge(cthb_cluster_month, left_index=True, right_index=True, validate='1:1')
    cthb_cluster = cthb_cluster.merge(cthb_cluster_event, left_index=True, right_index=True, validate='1:1')

    # Clustering
    n_clusters = 5
    km_cthb = KMeans(n_clusters = n_clusters, random_state=2).fit(cthb_cluster)
    cthb_cluster_label = km_cthb.labels_

    cthb_cluster_1 = cthb_cluster.iloc[cthb_cluster_label == 0, :]
    cthb_cluster_2 = cthb_cluster.iloc[cthb_cluster_label == 1, :]
    cthb_cluster_3 = cthb_cluster.iloc[cthb_cluster_label == 2, :]
    cthb_cluster_4 = cthb_cluster.iloc[cthb_cluster_label == 3, :]
    cthb_cluster_5 = cthb_cluster.iloc[cthb_cluster_label == 4, :]



    #################
    #  wif cluster #
    #################


    wif_cluster_temp = group_wif.T.merge(cal.set_index('date'), left_index=True, right_index=True, validate='1:1')
    wif_cluster_wday = wif_cluster_temp.groupby('wday').mean()[wif_cluster_temp.columns[:-13]].T
    wif_cluster_month = wif_cluster_temp.groupby('month').mean()[wif_cluster_temp.columns[:-13]].T
    wif_cluster_event = wif_cluster_temp.groupby('event_name_1').mean()[wif_cluster_temp.columns[:-13]].T[events]

    # normalize
    wif_cluster_event = wif_cluster_event.div(wif_cluster_wday.sum(axis=1), axis=0)
    wif_cluster_wday = wif_cluster_wday.div(wif_cluster_wday.sum(axis=1), axis=0)
    wif_cluster_month = wif_cluster_month.div(wif_cluster_month.sum(axis=1), axis=0)

    # Rename columns
    wif_cluster_wday = wif_cluster_wday.set_axis(wdays, axis='columns')
    wif_cluster_month = wif_cluster_month.set_axis(months, axis='columns')

    # Merge
    wif_cluster = wif_cluster_wday.merge(wif_cluster_month, left_index=True, right_index=True, validate='1:1')
    wif_cluster = wif_cluster.merge(wif_cluster_event, left_index=True, right_index=True, validate='1:1')

    # Clustering
    n_clusters = 5
    km_wif = KMeans(n_clusters = n_clusters, random_state=3).fit(wif_cluster)
    wif_cluster_label = km_wif.labels_

    wif_cluster_1 = wif_cluster.iloc[wif_cluster_label == 0, :]
    wif_cluster_2 = wif_cluster.iloc[wif_cluster_label == 1, :]
    wif_cluster_3 = wif_cluster.iloc[wif_cluster_label == 2, :]
    wif_cluster_4 = wif_cluster.iloc[wif_cluster_label == 3, :]
    wif_cluster_5 = wif_cluster.iloc[wif_cluster_label == 4, :]




    #################
    #  wihh cluster #
    #################


    wihh_cluster_temp = group_wihh.T.merge(cal.set_index('date'), left_index=True, right_index=True, validate='1:1')
    wihh_cluster_wday = wihh_cluster_temp.groupby('wday').mean()[wihh_cluster_temp.columns[:-13]].T
    wihh_cluster_month = wihh_cluster_temp.groupby('month').mean()[wihh_cluster_temp.columns[:-13]].T
    wihh_cluster_event = wihh_cluster_temp.groupby('event_name_1').mean()[wihh_cluster_temp.columns[:-13]].T[events]

    # normalize
    wihh_cluster_event = wihh_cluster_event.div(wihh_cluster_wday.sum(axis=1), axis=0)
    wihh_cluster_wday = wihh_cluster_wday.div(wihh_cluster_wday.sum(axis=1), axis=0)
    wihh_cluster_month = wihh_cluster_month.div(wihh_cluster_month.sum(axis=1), axis=0)

    # Rename columns
    wihh_cluster_wday = wihh_cluster_wday.set_axis(wdays, axis='columns')
    wihh_cluster_month = wihh_cluster_month.set_axis(months, axis='columns')

    # Merge
    wihh_cluster = wihh_cluster_wday.merge(wihh_cluster_month, left_index=True, right_index=True, validate='1:1')
    wihh_cluster = wihh_cluster.merge(wihh_cluster_event, left_index=True, right_index=True, validate='1:1')

    # Clustering
    n_clusters = 5
    km_wihh = KMeans(n_clusters = n_clusters, random_state=4).fit(wihh_cluster)
    wihh_cluster_label = km_wihh.labels_

    wihh_cluster_1 = wihh_cluster.iloc[wihh_cluster_label == 0, :]
    wihh_cluster_2 = wihh_cluster.iloc[wihh_cluster_label == 1, :]
    wihh_cluster_3 = wihh_cluster.iloc[wihh_cluster_label == 2, :]
    wihh_cluster_4 = wihh_cluster.iloc[wihh_cluster_label == 3, :]
    wihh_cluster_5 = wihh_cluster.iloc[wihh_cluster_label == 4, :]


    #################
    #  wihb cluster #
    #################


    wihb_cluster_temp = group_wihb.T.merge(cal.set_index('date'), left_index=True, right_index=True, validate='1:1')
    wihb_cluster_wday = wihb_cluster_temp.groupby('wday').mean()[wihb_cluster_temp.columns[:-13]].T
    wihb_cluster_month = wihb_cluster_temp.groupby('month').mean()[wihb_cluster_temp.columns[:-13]].T
    wihb_cluster_event = wihb_cluster_temp.groupby('event_name_1').mean()[wihb_cluster_temp.columns[:-13]].T[events]

    # normalize
    wihb_cluster_event = wihb_cluster_event.div(wihb_cluster_wday.sum(axis=1), axis=0)
    wihb_cluster_wday = wihb_cluster_wday.div(wihb_cluster_wday.sum(axis=1), axis=0)
    wihb_cluster_month = wihb_cluster_month.div(wihb_cluster_month.sum(axis=1), axis=0)

    # Rename columns
    wihb_cluster_wday = wihb_cluster_wday.set_axis(wdays, axis='columns')
    wihb_cluster_month = wihb_cluster_month.set_axis(months, axis='columns')

    # Merge
    wihb_cluster = wihb_cluster_wday.merge(wihb_cluster_month, left_index=True, right_index=True, validate='1:1')
    wihb_cluster = wihb_cluster.merge(wihb_cluster_event, left_index=True, right_index=True, validate='1:1')

    # Clustering
    n_clusters = 5
    km_wihb = KMeans(n_clusters = n_clusters, random_state=5).fit(wihb_cluster)
    wihb_cluster_label = km_wihb.labels_

    wihb_cluster_1 = wihb_cluster.iloc[wihb_cluster_label == 0, :]
    wihb_cluster_2 = wihb_cluster.iloc[wihb_cluster_label == 1, :]
    wihb_cluster_3 = wihb_cluster.iloc[wihb_cluster_label == 2, :]
    wihb_cluster_4 = wihb_cluster.iloc[wihb_cluster_label == 3, :]
    wihb_cluster_5 = wihb_cluster.iloc[wihb_cluster_label == 4, :]



    print(ctf_cluster_1.head())
    print(cthh_cluster_1.head())
    print(cthb_cluster_1.head())
    print(wif_cluster_1.head())
    print(wihh_cluster_1.head())
    print(wihb_cluster_1.head())


    ctf_index_1 = ctf_cluster_1.index.tolist()
    ctf_index_2 = ctf_cluster_2.index.tolist()
    ctf_index_3 = ctf_cluster_3.index.tolist()
    ctf_index_4 = ctf_cluster_4.index.tolist()
    ctf_index_5 = ctf_cluster_5.index.tolist()

    cthh_index_1 = cthh_cluster_1.index.tolist()
    cthh_index_2 = cthh_cluster_2.index.tolist()
    cthh_index_3 = cthh_cluster_3.index.tolist()
    cthh_index_4 = cthh_cluster_4.index.tolist()
    cthh_index_5 = cthh_cluster_5.index.tolist()

    cthb_index_1 = cthb_cluster_1.index.tolist()
    cthb_index_2 = cthb_cluster_2.index.tolist()
    cthb_index_3 = cthb_cluster_3.index.tolist()
    cthb_index_4 = cthb_cluster_4.index.tolist()
    cthb_index_5 = cthb_cluster_5.index.tolist()

    wif_index_1 = wif_cluster_1.index.tolist()
    wif_index_2 = wif_cluster_2.index.tolist()
    wif_index_3 = wif_cluster_3.index.tolist()
    wif_index_4 = wif_cluster_4.index.tolist()
    wif_index_5 = wif_cluster_5.index.tolist()

    wihh_index_1 = wihh_cluster_1.index.tolist()
    wihh_index_2 = wihh_cluster_2.index.tolist()
    wihh_index_3 = wihh_cluster_3.index.tolist()
    wihh_index_4 = wihh_cluster_4.index.tolist()
    wihh_index_5 = wihh_cluster_5.index.tolist()                
    
    wihb_index_1 = wihb_cluster_1.index.tolist()
    wihb_index_2 = wihb_cluster_2.index.tolist()
    wihb_index_3 = wihb_cluster_3.index.tolist()
    wihb_index_4 = wihb_cluster_4.index.tolist()
    wihb_index_5 = wihb_cluster_5.index.tolist()



    ctf_index = [ctf_index_1, ctf_index_2, ctf_index_3, ctf_index_4, ctf_index_5]
    cthh_index = [cthh_index_1, cthh_index_2, cthh_index_3, cthh_index_4, cthh_index_5]
    cthb_index = [cthb_index_1, cthb_index_2, cthb_index_3, cthb_index_4, cthb_index_5]
    wif_index = [wif_index_1, wif_index_2, wif_index_3, wif_index_4, wif_index_5]
    wihh_index = [wihh_index_1, wihh_index_2, wihh_index_3, wihh_index_4, wihh_index_5]
    wihb_index = [wihb_index_1, wihb_index_2, wihb_index_3, wihb_index_4, wihb_index_5]
    
    return ctf_index, cthh_index, cthb_index, wif_index, wihh_index, wihb_index


if __name__ == '__main__':
    _ = make_cluster()



    
