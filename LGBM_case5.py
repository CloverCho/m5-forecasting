import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import dask.dataframe as dd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import lightgbm as lgb
#import dask_xgboost as xgb
#import dask.dataframe as dd
from sklearn import preprocessing, metrics
from sklearn.preprocessing import LabelEncoder
import gc
import os
from tqdm import tqdm
from scipy.sparse import csr_matrix
import time







for dirname, _, filenames in os.walk('./data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns: #columns
        col_type = df[col].dtypes
        if col_type in numerics: #numerics
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def read_data():
    print('Reading files...')
    calendar = pd.read_csv('./data/calendar.csv')
    calendar = reduce_mem_usage(calendar)
    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))
    
    sell_prices = pd.read_csv('./data/sell_prices.csv')
    sell_prices = reduce_mem_usage(sell_prices)
    print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))
    
    #sales_train_val = pd.read_csv('./data/sales_train_validation.csv')
    sales_train_val = pd.read_csv('./data/modified_stv.csv')
    print('Sales train validation has {} rows and {} columns'.format(sales_train_val.shape[0], sales_train_val.shape[1]))
    
    submission = pd.read_csv('./data/sample_submission.csv')
    
    return calendar, sell_prices, sales_train_val, submission



import IPython

def display(*dfs, head=True):
    for df in dfs:
        IPython.display.display(df.head() if head else df)


def encode_categorical(df, cols):
    
    for col in cols:
        # Leave NaN as it is.
        le = LabelEncoder()
        #not_null = df[col][df[col].notnull()]
        df[col] = df[col].fillna('nan')
        df[col] = pd.Series(le.fit_transform(df[col]), index=df.index)

    return df


def simple_fe(data):
    
    # demand features
    
    for diff in [0, 1, 2, 3, 4, 5, 6]:
        shift = DAYS_PRED + diff
        data[f"shift_t{shift}"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(shift)
        )
    
    
    
    for size in [7, 30]:
        data[f"rolling_mean_t{size}"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(DAYS_PRED).rolling(size).mean()
        )
    
    
    # time features
    dt_col = "date"
    data[dt_col] = pd.to_datetime(data[dt_col])
    


    attrs = [
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "dayofweek",
        "is_year_end",
        "is_year_start",
        "is_quarter_end",
        "is_quarter_start",
        "is_month_end",
        "is_month_start",
    ]

    for attr in attrs:
        dtype = np.int16 if attr == "year" else np.int8
        data[attr] = getattr(data[dt_col].dt, attr).astype(dtype)

    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(np.int8)
    
    return data



calendar, sell_prices, sales_train_val, submission = read_data()

NUM_ITEMS = sales_train_val.shape[0]
DAYS_PRED = submission.shape[1] - 1
print(NUM_ITEMS, DAYS_PRED)


calendar = encode_categorical(calendar, ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]).pipe(reduce_mem_usage)

sales_train_val = encode_categorical(sales_train_val, ["item_id", "dept_id", "cat_id", "store_id", "state_id", "clusterID"],).pipe(reduce_mem_usage)

sell_prices = encode_categorical(sell_prices, ["item_id", "store_id"]).pipe(reduce_mem_usage)

product = sales_train_val[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'clusterID']].drop_duplicates()


nrows = 365 * 2 * NUM_ITEMS

print(sales_train_val.head(5))

d_name = ['d_' + str(i+1) for i in range(1913)]

sales_train_val_values = sales_train_val[d_name].values

# calculate the start position(first non-zero demand observed date) for each item 
tmp = np.tile(np.arange(1,1914),(sales_train_val_values.shape[0],1))
df_tmp = ((sales_train_val_values>0) * tmp)

start_no = np.min(np.where(df_tmp==0,9999,df_tmp),axis=1)-1

flag = np.dot(np.diag(1/(start_no+1)) , tmp)<1

sales_train_val_values = np.where(flag,np.nan,sales_train_val_values)

sales_train_val[d_name] = sales_train_val_values

del tmp,sales_train_val_values
gc.collect()


print(1913-np.max(start_no))

sales_train_val = pd.melt(sales_train_val,id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'clusterID'], var_name = 'day', value_name = 'demand')

print(sales_train_val.head(5))
print('Melted sales train validation has {} rows and {} columns'.format(sales_train_val.shape[0],sales_train_val.shape[1]))

sales_train_val = sales_train_val.iloc[-nrows:,:]
sales_train_val = sales_train_val[~sales_train_val.demand.isnull()]

test1_rows = [row for row in submission['id'] if 'validation' in row]
test2_rows = [row for row in submission['id'] if 'evaluation' in row]


test1 = submission[submission['id'].isin(test1_rows)]
test2 = submission[submission['id'].isin(test2_rows)]



test1.columns = ["id"] + [f"d_{d}" for d in range(1914, 1914 + DAYS_PRED)]
test2.columns = ["id"] + [f"d_{d}" for d in range(1942, 1942 + DAYS_PRED)]


#test1['id'] = test1['id'].str.replace('_validation','')
test2['id'] = test2['id'].str.replace('_evaluation','_validation')



test1 = test1.merge(product, how = 'left', on = 'id')
test2 = test2.merge(product, how = 'left', on = 'id')


test1 = pd.melt(test1, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'clusterID'], var_name = 'day', value_name = 'demand')
test2 = pd.melt(test2, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'clusterID'], var_name = 'day', value_name = 'demand')


sales_train_val['part'] = 'train'
test1['part'] = 'test1'
test2['part'] = 'test2'


data = pd.concat([sales_train_val, test1, test2], axis = 0)

del sales_train_val, test1, test2

data = data[data['part'] != 'test2']

gc.collect()



calendar.drop(['weekday', 'wday', 'month', 'year'], 
            inplace = True, axis = 1)


data = pd.merge(data, calendar, how = 'left', left_on = ['day'], right_on = ['d'])
data.drop(['d', 'day'], inplace = True, axis = 1)


del  calendar
gc.collect()

# get the sell price data (this feature should be very important)
data = data.merge(sell_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
print('Our final dataset to train has {} rows and {} columns'.format(data.shape[0], data.shape[1]))


del  sell_prices
gc.collect()

print(data.head(3))


data = simple_fe(data)
data = reduce_mem_usage(data)

print(data.head())



# going to evaluate with the last 28 days
x_train = data[data['date'] <= '2016-03-27']
y_train = x_train['demand']
x_val = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
y_val = x_val['demand']
test = data[(data['date'] > '2016-04-24')]

features = [
    "item_id",
    "store_id",
    "clusterID",
    "shift_t28",
    "shift_t29",
    "shift_t30",
    "shift_t31",
    "shift_t32",
    "shift_t33",
    "shift_t34",
    "sell_price",
    "year",
    "month",
    "week",
    "dayofweek",
    "rolling_mean_t7",
    "rolling_mean_t30",
]





train_set = lgb.Dataset(x_train[features], y_train)
val_set = lgb.Dataset(x_val[features], y_val)

del x_train, y_train

'''
# model estimation
model = lgb.train(params, train_set, num_boost_round = 2500, early_stopping_rounds = 50, valid_sets = [train_set, val_set], verbose_eval = 100)
val_pred = model.predict(x_val[features])
val_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
print(f'Our val rmse score is {val_score}')
y_pred = model.predict(test[features])
test['demand'] = y_pred
'''    


'''
predictions = test[['id', 'date', 'demand']]
predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()
predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

evaluation_rows = [row for row in submission['id'] if 'evaluation' in row] 
evaluation = submission[submission['id'].isin(evaluation_rows)]

validation = submission[['id']].merge(predictions, on = 'id')
final = pd.concat([validation, evaluation])
final.to_csv('submission.csv', index = False)
'''

weight_mat = np.c_[np.ones([NUM_ITEMS,1]).astype(np.int8), # level 1
                pd.get_dummies(product.state_id.astype(str),drop_first=False).astype('int8').values,
                pd.get_dummies(product.store_id.astype(str),drop_first=False).astype('int8').values,
                pd.get_dummies(product.cat_id.astype(str),drop_first=False).astype('int8').values,
                pd.get_dummies(product.dept_id.astype(str),drop_first=False).astype('int8').values,
                pd.get_dummies(product.state_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
                pd.get_dummies(product.state_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
                pd.get_dummies(product.store_id.astype(str) + product.cat_id.astype(str),drop_first=False).astype('int8').values,
                pd.get_dummies(product.store_id.astype(str) + product.dept_id.astype(str),drop_first=False).astype('int8').values,
                pd.get_dummies(product.item_id.astype(str),drop_first=False).astype('int8').values,
                pd.get_dummies(product.state_id.astype(str) + product.item_id.astype(str),drop_first=False).astype('int8').values,
                np.identity(NUM_ITEMS).astype(np.int8) #item :level 12
                ].T

weight_mat_csr = csr_matrix(weight_mat)
del weight_mat; gc.collect()



def weight_calc(data,product):
    
    # calculate the denominator of RMSSE, and calculate the weight base on sales amount

    sales_train_val = pd.read_csv('./data/modified_stv.csv')

    d_name = ['d_' + str(i+1) for i in range(1913)]

    sales_train_val = weight_mat_csr * sales_train_val[d_name].values

    # calculate the start position(first non-zero demand observed date) for each item 
    df_tmp = ((sales_train_val>0) * np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))

    start_no = np.min(np.where(df_tmp==0,9999,df_tmp),axis=1)-1

    flag = np.dot(np.diag(1/(start_no+1)) , np.tile(np.arange(1,1914),(weight_mat_csr.shape[0],1)))<1

    sales_train_val = np.where(flag,np.nan,sales_train_val)

    # denominator of RMSSE / RMSSE
    weight1 = np.nansum(np.diff(sales_train_val,axis=1)**2,axis=1)/(1913-start_no)

    # calculate the sales amount for each item/level
    df_tmp = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]
    df_tmp['amount'] = df_tmp['demand'] * df_tmp['sell_price']
    df_tmp =df_tmp.groupby(['id'])['amount'].apply(np.sum)
    df_tmp = df_tmp[product.id].values
    
    weight2 = weight_mat_csr * df_tmp 

    weight2 = weight2/np.sum(weight2)

    del sales_train_val
    gc.collect()
    
    return weight1, weight2

weight1, weight2 = weight_calc(data,product)

def wrmsse(preds, data):
    
    # this function is calculate for last 28 days to consider the non-zero demand period
    
    # actual obserbed values 
    y_true = data.get_label()
    
    y_true = y_true[-(NUM_ITEMS * DAYS_PRED):]
    preds = preds[-(NUM_ITEMS * DAYS_PRED):]
    # number of columns
    num_col = DAYS_PRED
    
    # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) 
    reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
    reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T
    
          
    train = weight_mat_csr*np.c_[reshaped_preds, reshaped_true]
    
    score = np.sum(
                np.sqrt(
                    np.mean(
                        np.square(
                            train[:,:num_col] - train[:,num_col:])
                        ,axis=1) / weight1) * weight2)
    
    return 'wrmsse', score, False

def wrmsse_simple(preds, data):
    
    # actual obserbed values 
    y_true = data.get_label()
    
    y_true = y_true[-(NUM_ITEMS * DAYS_PRED):]
    preds = preds[-(NUM_ITEMS * DAYS_PRED):]
    # number of columns
    num_col = DAYS_PRED
    
    # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) )
    reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
    reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T
          
    train = np.c_[reshaped_preds, reshaped_true]
    
    weight2_2 = weight2[:NUM_ITEMS]
    weight2_2 = weight2_2/np.sum(weight2_2)
    
    score = np.sum(
                np.sqrt(
                    np.mean(
                        np.square(
                            train[:,:num_col] - train[:,num_col:])
                        ,axis=1) /  weight1[:NUM_ITEMS])*weight2_2)
    
    return 'wrmsse', score, False



params = {
    'boosting_type': 'gbdt',
    'metric': 'custom',
    'objective': 'poisson',
    'n_jobs': -1,
    'seed': 236,
    'learning_rate': 0.1,
    'bagging_fraction': 0.75,
    'bagging_freq': 10, 
    'colsample_bytree': 0.75}

# model estimation
start_time = time.time()

model = lgb.train(params, train_set, num_boost_round = 2500, early_stopping_rounds = 50, 
                valid_sets = [train_set, val_set], verbose_eval = 100, feval= wrmsse)

finish_time = time.time()
print("Train time: {} seconds".format(finish_time - start_time))


val_pred = model.predict(x_val[features])
val_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))
print(f'Our val wrmsse score is {val_score}')


y_pred = model.predict(test[features])
test['demand'] = y_pred


predictions = test[['id', 'date', 'demand']]
predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'demand').reset_index()
predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]

evaluation_rows = [row for row in submission['id'] if 'evaluation' in row] 
evaluation = submission[submission['id'].isin(evaluation_rows)]

validation = submission[['id']].merge(predictions, on = 'id')
final = pd.concat([validation, evaluation])




final.to_csv('submission_CHOandSON_LGBM5.csv', index = False)