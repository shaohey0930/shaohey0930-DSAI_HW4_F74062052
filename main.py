# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 01:19:20 2021

@author: h8273
"""
import pandas as pd
import numpy as np
import datetime
# import matplotlib.pyplot as plt
# import seaborn as sns
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout

# Import all of them 
sales=pd.read_csv("sales_train.csv")
item_cat=pd.read_csv("item_categories.csv")
item=pd.read_csv("items.csv")
sub=pd.read_csv("sample_submission.csv")
shops=pd.read_csv("shops.csv")
test=pd.read_csv("test.csv")

#formatting the date column correctly
sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
# check
print(sales.info())

## Data Cleaning

# clean data if item price is not > 0
print('item price < 0: ', sales[sales.item_price < 0])
sales = sales.query('item_price > 0')

# delete data whose item & shop don't exist in test data
print('test shop id: ',test['shop_id'].unique())
sales = sales[sales['shop_id'].isin(test['shop_id'].unique())]

print('test item id: ',test['item_id'].unique())
sales = sales[sales['item_id'].isin(test['item_id'].unique())]

# delete outliers whose item_cnt_day > 200 & item_price > 50000
print('outliers: ', sales[sales.item_cnt_day > 200])
print('outliers: ', sales[sales.item_price > 50000])



## Data Preprocessing

monthly_processed_sales=sales.groupby(["date_block_num","shop_id","item_id"])[
    "date_block_num","date","item_price","item_cnt_day"].agg({"date_block_num":'mean',"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})

# extract the test data column from processed data
monthly_sales_flat = monthly_processed_sales.item_cnt_day.apply(list).reset_index()

# merge the test data with processed data
merged_monthly_sales_flat = pd.merge(test, monthly_sales_flat, on = ['item_id','shop_id'], how = 'left')

# fill nan with 0
merged_monthly_sales_flat.fillna(0, inplace= True)

# drop the data columns doesn't exist in test data
merged_monthly_sales_flat.drop(['shop_id','item_id'],inplace = True, axis = 1)

# create train data table for LSTM
train_data_table = merged_monthly_sales_flat.pivot_table(index = 'ID', columns='date_block_num', fill_value=0, aggfunc = 'sum')

x_train = np.expand_dims(train_data_table.values[:,:-1], axis = 2)
y_train = train_data_table.values[:, -1:]
x_test = np.expand_dims(train_data_table.values[:, 1:] , axis = 2)

print(x_train.shape,y_train.shape,x_test.shape)

# ARIMA Model
# import pmdarima as pm
# import numpy as np
# import pandas as pd
# from pmdarima.arima import ndiffs

# arima_output = np.zeros((214200, 1))

# for i in range(214200):
#     # load data
#     train = x_test[i].flatten()
    
#     # choose diffencing
    
#     # Fit a simple auto_arima model
#     auto = pm.auto_arima(train, d= 1, seasonal=True, stepwise=True,
#                          suppress_warnings=True, error_action="ignore", max_p=6,
#                          max_order=None, trace=True)
    
#     # predict with range of 3/22 to 3/29
#     fc, conf_int = auto.predict(n_periods=1, return_conf_int=True)
#     arima[i] = fc[0]



## LSTM Model

# model define
model = Sequential()
model.add(LSTM(units = 64,input_shape = (33,1)))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])
model.summary()

# fit the data and predict
model.fit(x_train,y_train,batch_size = 4096,epochs = 10)
output = model.predict(x_test)

# merge and create submission.csv

submission = pd.DataFrame({'ID':test['ID'],'item_cnt_month':output.ravel()})
submission.to_csv('submission.csv',index = False)


