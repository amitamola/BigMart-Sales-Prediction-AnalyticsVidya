import pandas as pd
import numpy as np
import xgboost as xgb

train_data_df = pd.read_csv('train.csv',delimiter=',',header = None)
test_data_df = pd.read_csv('test.csv',header = None ,delimiter=",")

train_data_df.columns = ['Item_Weight','Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Outlet_Sales']
test_data_df.columns = ['Item_Weight','Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type']

myResults = train_data_df['Item_Outlet_Sales'] 
myResults = np.array(myResults)

labels_numeric = pd.Series(train_data_df['Item_Outlet_Sales'],dtype = "float")
labels_numeric = labels_numeric.astype(np.float)
#print labels_numeric
train_data_df = train_data_df.drop('Item_Outlet_Sales',1)

train_data_df = np.array(train_data_df)

test_data_df = np.array(test_data_df)

xg_train = xgb.DMatrix(train_data_df,label=labels_numeric)
xg_test = xgb.DMatrix(test_data_df)

param = {}
# param['eval_metric'] = 'rmse'
# param['objective'] = 'reg:linear'
# param['booster'] = 'gblinear' 
param['eta'] = 0.1
param['gamma'] = 1
# param['n_estimators'] = 1000
param['min_child_weight'] = 5
param['max_depth'] = 8
# param['subsample'] = 0.5
# param['colsample_bytree'] = 0.5
# param['max_delta_step'] = 20
# param['lambda'] = 10
num_round = 1000

gbm = xgb.train(param,xg_train,num_round)
test_pred = gbm.predict(xg_test,output_margin = True)

f1 = open('id.csv')
id_list = []
for i in f1 :
	j = i.strip()
	id_list.append(j)

f = open('results.csv','w')
f.write("Item_Identifier,Outlet_Identifier,Item_Outlet_Sales\n")

for i in range(len(test_pred)) :
	j = str(test_pred[i])
	f.write(str(id_list[i])+","+str(j)+"\n")
