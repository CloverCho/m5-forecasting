import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from itertools import cycle
pd.set_option('max_columns', 50)
plt.style.use('bmh')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

INPUT_DIR = '../../data'
cal = pd.read_csv(f'{INPUT_DIR}/calendar.csv')
stv= pd.read_csv(f'{INPUT_DIR}/sales_train_validation.csv')
ss = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
sellp = pd.read_csv(f'{INPUT_DIR}/sell_prices.csv')


d_cols = [c for c in stv.columns if 'd_' in c] 

# 1. Count of Items by State
stv.groupby('state_id').count()['id'] \
    .sort_values() \
    .plot(kind='barh', figsize=(15,5), title='Count of Items by State')

plt.show()

'''
# 2. Rolling 28 Day Average Total Sales
past_sales = stv.set_index('id')[d_cols].T.merge(cal.set_index('d'), left_index=True, right_index=True, validate='1:1').set_index('date')

store_list = stv['store_id'].unique()

for store in store_list:
    store_items = [c for c in past_sales.columns if store in c]
    past_sales[store_items].sum(axis=1).rolling(28).mean().plot(figsize=(18,6), alpha=0.8)

plt.title('Rolling 28 Day Average Total Sales (10 stores)', size = 20)
plt.xticks(size=13)
plt.yticks(size=13)
plt.xlabel('Date', size = 15)
plt.ylabel('Sales', size=15)
plt.legend(store_list)
plt.show()
'''

# 3. 
'''


print(past_sales.head())


for i in stv['cat_id'].unique():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,3))
    items_col = [c for c in past_sales.columns if i in c]
    past_sales.groupby('wday').mean()[items_col].sum(axis=1) \
        .plot(kind='line',
              title='Average sale: day of week',
              lw=5,
              color=color_pal[0],
              ax = ax1)

    past_sales.groupby('month').mean()[items_col].sum(axis=1) \
        .plot(kind='line',
              title='Average sale: month',
              lw=5,
              color=color_pal[3],
              ax = ax2)

    past_sales.groupby('year').mean()[items_col].sum(axis=1) \
        .plot(kind='line',
              title='Average sale: year',
              lw=5,
              color=color_pal[6],
              ax = ax3)                  
    
    fig.suptitle(f'Trends for {i}', size=20)
    plt.tight_layout()
    plt.show()


'''

#4. Trends for random items

example = stv.loc[stv['id'] == 'FOODS_3_131_CA_3_validation'][d_cols].T
print(example.head())
example = example.rename(columns={8453:'FOODS_3_131_CA_3'}) # Name it correctly
example = example.reset_index().rename(columns={'index': 'd'}) # make the index 'd'
example = example.merge(cal, how='left', validate='1:1')
example.set_index('date')['FOODS_3_131_CA_3'] \
    .plot(figsize=(15, 5),
          color=next(color_cycle),
          title='FOODS_3_131_CA_3 sales by actual sale dates')

plt.show()


example2 = stv.loc[stv['id'] == 'FOODS_3_085_CA_3_validation'][d_cols].T
print(example2.head())

example2 = example2.rename(columns={8407:'FOODS_3_085_CA_3'}) # Name it correctly
example2 = example2.reset_index().rename(columns={'index': 'd'}) # make the index 'd'
example2 = example2.merge(cal, how='left', validate='1:1')
example2.set_index('date')['FOODS_3_085_CA_3'] \
    .plot(figsize=(15, 5),
          color=next(color_cycle),
          title='FOODS_3_085_CA_3 sales by actual sale dates')

plt.show()


example3 = stv.loc[stv['id'] == 'FOODS_3_007_CA_3_validation'][d_cols].T
print(example3.head())

example3 = example3.rename(columns={8330:'FOODS_3_007_CA_3'}) # Name it correctly
example3 = example3.reset_index().rename(columns={'index': 'd'}) # make the index 'd'
example3 = example3.merge(cal, how='left', validate='1:1')
example3.set_index('date')['FOODS_3_007_CA_3'] \
    .plot(figsize=(15, 5),
          color=next(color_cycle),
          title='FOODS_3_007_CA_3 sales by actual sale dates')

plt.show()

examples = ['FOODS_3_131_CA_3', 'FOODS_3_085_CA_3', 'FOODS_3_007_CA_3']


example_df = [example, example2, example3]
for i in [0, 1, 2]:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,3))
    example_df[i].groupby('wday').mean()[examples[i]] \
        .plot(kind='line',
              title='average sale: day of week',
              lw=5,
              color=color_pal[0],
              ax=ax1)

    example_df[i].groupby('month').mean()[examples[i]] \
        .plot(kind='line',
              title='average sale: month',
              lw=5,
              color=color_pal[4],
              ax=ax2)
    
    '''
    example_df[i].groupby('year').mean()[examples[i]] \
        .plot(kind='line',
              title='average sale: year',
              color=color_pal[2],
              lw=5,
              ax=ax3)
    '''
    fig.suptitle(f'Trends for item: {examples[i]}', size=20)
    plt.tight_layout()
    plt.show()

'''

#5. Trends for States


for i in stv['state_id'].unique():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,3))
    items_col = [c for c in past_sales.columns if i in c]
    past_sales.groupby('wday').mean()[items_col].sum(axis=1) \
        .plot(kind='line',
              title='Average sale: day of week',
              lw=5,
              color=color_pal[0],
              ax = ax1)

    past_sales.groupby('month').mean()[items_col].sum(axis=1) \
        .plot(kind='line',
              title='Average sale: month',
              lw=5,
              color=color_pal[3],
              ax = ax2)

    past_sales.groupby('year').mean()[items_col].sum(axis=1) \
        .plot(kind='line',
              title='Average sale: year',
              lw=5,
              color=color_pal[6],
              ax = ax3)                  
    
    fig.suptitle(f'Trends for {i}', size=20)
    plt.tight_layout()
    plt.show()


# 6. Trends for Stores

for i in stv['store_id'].unique():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,3))
    items_col = [c for c in past_sales.columns if i in c]
    past_sales.groupby('wday').mean()[items_col].sum(axis=1) \
        .plot(kind='line',
              #title='Average sale: day of week',
              lw=5,
              color=color_pal[0],
              ax = ax1)

    past_sales.groupby('month').mean()[items_col].sum(axis=1) \
        .plot(kind='line',
              #title='Average sale: month',
              lw=5,
              color=color_pal[3],
              ax = ax2)

    past_sales.groupby('year').mean()[items_col].sum(axis=1) \
        .plot(kind='line',
              #title='Average sale: year',
              lw=5,
              color=color_pal[6],
              ax = ax3)                  
    
    fig.suptitle(f'{i}', size=20)
    plt.tight_layout()
    plt.show()
'''

'''
    # 5. store - date
store_CA = ['CA_1', 'CA_2', 'CA_3', 'CA_4']
store_TX = ['TX_1', 'TX_2', 'TX_3']
store_WI = ['WI_1', 'WI_2', 'WI_3']

fig, axes = plt.subplots(4, 1, figsize=(12, 5), sharex=True)
ax_idx = 0

for ca in store_CA:
    items_col = [c for c in past_sales.columns if ca in c]
    past_sales[items_col].sum(axis=1) \
        .plot(alpha=1,
              lw=1,
              color=next(color_cycle),
              title=ca,
              ax = axes[ax_idx])
    ax_idx += 1

plt.tight_layout()               
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True)
ax_idx = 0

for tx in store_TX:
    items_col = [c for c in past_sales.columns if tx in c]
    past_sales[items_col].sum(axis=1) \
        .plot(alpha=1,
              lw=1,
              color=next(color_cycle),
              title=tx,
              ax = axes[ax_idx])
    ax_idx += 1

plt.tight_layout()               
plt.show()


fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True)
ax_idx = 0

for wi in store_WI:
    items_col = [c for c in past_sales.columns if wi in c]
    past_sales[items_col].sum(axis=1) \
        .plot(alpha=1,
              lw=1,
              color=next(color_cycle),
              title=wi,
              ax = axes[ax_idx])
    ax_idx += 1

plt.tight_layout()               
plt.show()


#6. snap

fig, axs = plt.subplots(3, 3, figsize=(20, 16))

states = ['CA', 'TX', 'WI']
categories = ['FOODS', 'HOBBIES', 'HOUSEHOLD']
date_range = cal['date']
date_cols = [c for c in stv.columns if 'd_' in c]



for i, state in enumerate(states):
    snap_id = 'snap_%s' % state
    snap_state = pd.Series(cal[snap_id][:len(date_cols)])
    snap_state_df = snap_state.reset_index()
    print(snap_state_df.head())
    snap_on_state_df = snap_state_df[snap_state_df[snap_id] == 1]
    snap_off_state_df = snap_state_df[snap_state_df[snap_id] == 0]

    state_df = stv[stv['state_id'] == state]
    for j, cat in enumerate(categories):
        dept_wise_df = state_df[state_df['cat_id'] == cat]
        aggr_array = []
        for d in date_cols:
            aggr_array.append(dept_wise_df[d].values.sum())
        daily_time_series_df = pd.DataFrame(data=aggr_array, columns=['Sales'], index=date_cols)
        series = daily_time_series_df['Sales']

        X_values = range(len(snap_off_state_df.index.values))
        coeffs = np.polyfit(X_values, daily_time_series_df['Sales'].values[snap_off_state_df.index.values], 7)
        poly_eqn = np.poly1d(coeffs)
        y_hat_snap_off = poly_eqn(X_values)

        X_values = range(len(snap_on_state_df.index.values))
        coeffs = np.polyfit(X_values, daily_time_series_df['Sales'].values[snap_on_state_df.index.values], 7)
        poly_eqn = np.poly1d(coeffs)
        y_hat_snap_on = poly_eqn(X_values)

        axs[i, j].plot(snap_off_state_df['index'], y_hat_snap_off, label='non snap', linewidth=2, color = 'green')
        axs[i, j].plot(snap_on_state_df['index'], y_hat_snap_on, label='with snap', linewidth=2, color='red')
        axs[i, j].plot(snap_off_state_df['index'], daily_time_series_df['Sales'].values[snap_off_state_df.index.values],
                                                    linewidth=2, color='green', alpha=0.2)

        axs[i, j].plot(snap_on_state_df['index'], daily_time_series_df['Sales'].values[snap_on_state_df.index.values],
                                                    linewidth=2, color='red', alpha=0.1)
        
        axs[i, j].legend()
        axs[i, j].set_title('%s - sales - %s' % (cat, state), fontsize=18)
        axs[i, j].set(xlabel='Date', ylabel='Sales Units')


plt.tight_layout()
#plt.savefig('../output/snap - sate_cat.png', dpi=300)
plt.show()


# Holidays

data_means = pd.DataFrame(stv.iloc[:, 6:].mean(), columns=['mean'])


holidays = cal[['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']]
uholidays = pd.unique(holidays[['event_name_1', 'event_name_2']].values.ravel())[1:]
holidays_shifted = holidays.shift(-14).loc[:data_means.shape[0] - 1, :]

print(uholidays)

# aligh dates to each holiday
for holiday in uholidays:
    dayno = 0
    daynos = []
    for index, row in holidays_shifted.iterrows():
        if dayno > 0:
            dayno += 1
        if dayno > 21:
            dayno = 0
        if (row.event_name_1 == holiday) | (row.event_name_2 == holiday):
            dayno = 1
        daynos.append(dayno)
    
    data_means['dayno'] = daynos
    data_means['dayno'] -= 15
    df = data_means.groupby('dayno', sort=True).mean()
    df.columns = [holiday]
    df['ref'] = df.iloc[0, 0]
    ax = df[1:].plot(figsize=(15,4))
    ax.locator_params(integer=True)
    ax.axvline(x=0)
    plt.title('{}'.format(holiday), fontsize=18)
    plt.savefig('../output/event_{}.png'.format(holiday), dpi=300)
    plt.show()

'''
























