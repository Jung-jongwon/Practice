#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import sys
import seaborn
import matplotlib.pyplot as plt
# example of training a final classification model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras.models import load_model


# In[21]:


import tensorflow as tf
print( f'tensorflow_ver : {tf.__version__}')
import keras
print(f'keras_ver : {keras.__version__}')
import pandas
print(f'pandas_ver : {pandas.__version__}')
import sklearn
print(f'sklearn_ver : {sklearn.__version__}')
import sys
print( f'python_ver : {sys.version[0:5]}' )


# In[84]:


data_20s = pd.read_csv('data_mapping_20s_0112_change.csv')
data_30s = pd.read_csv('data_mapping_30s_0112_change.csv')
data_40s = pd.read_csv('data_mapping_40s_0112_change.csv')
data_50s = pd.read_csv('data_mapping_50s_0112_change.csv')
data_60s = pd.read_csv('data_mapping_60s_0112_change.csv')
data_70s = pd.read_csv('data_mapping_70s_0112_change.csv')
data_80s = pd.read_csv('data_mapping_80s_0112_change.csv')


# In[85]:


data_full = data_20s.append(data_30s)
data_full = data_full.append(data_40s)
data_full = data_full.append(data_50s)
data_full = data_full.append(data_60s)
data_full = data_full.append(data_70s)
data_full = data_full.append(data_80s)


# In[86]:


data_full = data_full.reset_index()
data_full=data_full[['HR','RR','HRV','SDNN','RMSSD','PNN50','gender','age','BP_HIGH']]


# In[87]:


data_full


# In[88]:


data_full[data_full['SDNN']<1].index


# In[89]:


del_index=[]
for i in range(0,len(data_full)):
    if data_full.loc[i,'SDNN']<1:
        del_index.append(i)


# In[90]:


data_full = data_full.drop(index=del_index)
data_full = data_full.reset_index()
data_full=data_full[['HR','RR','HRV','SDNN','RMSSD','PNN50','gender','age','BP_HIGH']]


# In[91]:


data_full


# In[92]:


data_full[data_full['BP_HIGH']>300].index


# In[93]:


del_index=[]
for i in range(0,len(data_full)):
    if data_full.loc[i,'BP_HIGH']>300:
        del_index.append(i)


# In[70]:


data_full = data_full.drop(index=del_index)
data_full = data_full.reset_index()
data_full=data_full[['HR','RR','HRV','SDNN','RMSSD','PNN50','gender','age','BP_HIGH']]


# In[71]:


data_full


# In[ ]:





# In[ ]:





# In[ ]:





# In[72]:


raw = pd.read_csv('rawdata_chosun_2.csv')


# In[73]:


raw


# In[74]:


data_full = data_full.append(raw)


# In[75]:


data_full=data_full[['HR','RR','HRV','SDNN','RMSSD','PNN50','gender','age','BP_HIGH']]
data_full


# In[76]:


data_full = data_full.dropna(axis=0)
data_full = data_full.reset_index()
data_full=data_full[['HR','RR','HRV','SDNN','RMSSD','PNN50','gender','age','BP_HIGH']]
data_full


# In[77]:


data_full[data_full['BP_HIGH']>300].index


# In[78]:


del_index=[]
for i in range(0,len(data_full)):
    if data_full.loc[i,'BP_HIGH']>300:
        del_index.append(i)


# In[79]:


data_full = data_full.drop(index=del_index)
data_full = data_full.reset_index()
data_full=data_full[['HR','RR','HRV','SDNN','RMSSD','PNN50','gender','age','BP_HIGH']]
data_full


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[80]:


# Reading the traing file
# df = pd.read_csv('../03. 개발 및 검증 데이터/BP_train_20210913.csv')
df = data_full
result_df = df
# 실제 데이터
# del df['age']
# del df['agegroup']
# del df['bloodcategory']
# del df['DM']

# Read the verification file
df1 = pd.read_csv('BP_test_20210913.csv')
result_df1 = df1
# del df1['age']
# del df1['agegroup']
# del df1['bloodcategory']
# del df1['DM']

# df2 = df[df['DM'] == 0]


df.columns
df


X = df.drop('BP_HIGH',axis=1)
print(X)
y = df['BP_HIGH']
print("==============================")

X1 = df.drop('BP_HIGH',axis=1)
y1 = df['BP_HIGH']

X2 = df1.drop('BP_HIGH', axis=1)
y2 = df1['BP_HIGH']




X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=30)


# input scaling
# scaler = RobustScaler()
# scaler = MinMaxScaler()
scaler = StandardScaler()
# StandardScaler standardization file SAVE!!!!
X_train.to_csv('BP_HIGH_0112_delbp300.csv')

X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# scale 변환
X1 = scaler.transform(X1)

print(X_train.shape)
print(X_test.shape)






#  (20210717) 과적합을 해결하기 위해 
#  kernel_regularizer, activity_regularizer, Dropout 을 적용하였다
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer='he_normal',bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.001)))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(380, activation=tf.nn.relu, kernel_initializer='he_normal',bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.001)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(320, activation=tf.nn.relu, kernel_initializer='he_normal',bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.001)))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
model.add(tf.keras.layers.Dense(192, activation=tf.nn.relu, kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu, kernel_initializer='normal'))
model.add(tf.keras.layers.Dense(1, kernel_initializer='zeros', activation='linear'))
# opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile( loss='mse',  optimizer=Adam(learning_rate=0.0005,beta_1=0.99), metrics=['accuracy'])


# fit model
history = model.fit(X_train, y_train.values, epochs=200, batch_size=40, validation_split = 0.25, verbose=1)


## Save the model to the computer - check the path
# Model 저장 
model.save('BP_HIGH_0112_delbp300.h5')


## plot setting 
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)




def print_evaluate(true, predicted, train=True):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    if train:
        print("========Training Result=======")
        print('MAE: ', mae)
        print('MSE: ', mse)
        print('RMSE: ', rmse)
        print('R2 score: ', r2_square)
    elif not train:
        print("=========Testing Result=======")
        print('MAE: ', mae)
        print('MSE: ', mse)
        print('RMSE: ', rmse)
        print('R2 score: ', r2_square)



y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Training file scaler application(X1 predict )
y1_predict = model.predict(X1)


# Apply test file scaler(X2 predict)
X2 = scaler.transform(X2)
y2_predict = model.predict(X2)

print_evaluate(y_train, y_train_pred, train=True)
print_evaluate(y_test, y_test_pred, train=False)

print("================== result random reference ================")
print("================== result random reference ================")
print_evaluate(y1, y1_predict, train=False)

print("================== result random reference 2000 ================")
print("================== result random reference 2000 ================")
print_evaluate(y2, y2_predict, train=False)


result_df['predict_result'] = y1_predict
result_df.to_csv(r"result_tr_01.csv")
print(y1_predict)

result_df1['predict_result'] = y2_predict
result_df1.to_csv(r"result1_vd_01.csv")
print(y2_predict)


#======================================================
# Comparison of predicted and actual values
testY2_predict = model.predict(X2)
for i in range(len(testY2_predict)):
    label = y2[i]
    prediction = testY2_predict[i]
    print('RealSugar| {:} | PredictSugar| {:}'.format(label, prediction))
#======================================================



# ====================Scatter Plot==================
# draw scatter plot
plt.title("Scatter Plot", fontsize=15)
plt.scatter(y1, y1_predict, color='black', alpha=.5)
plt.xlabel("BP of reference", fontsize=13)
plt.ylabel("BP of AI", fontsize=13)
plt.grid()
plt.show()
# ====================Scatter Plot==================



import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
from joblib import dump, load


dump(scaler, 'BP_HIGH_0112_delbp300.bin', compress=True)


# In[5]:


def print_evaluate(true, predicted, train=True):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    if train:
        print("========Training Result=======")
        print('MAE: ', mae)
        print('MSE: ', mse)
        print('RMSE: ', rmse)
        print('R2 score: ', r2_square)
    elif not train:
        print("=========Testing Result=======")
        print('MAE: ', mae)
        print('MSE: ', mse)
        print('RMSE: ', rmse)
        print('R2 score: ', r2_square)





# ### Import data with StandardScaler standardization work
dm_standard_df = pd.read_csv( 'BloodPressure_high(ver1.3)_0930_01.csv' )
# print(dm_standard_df)
standard_result_df = dm_standard_df
standardX = dm_standard_df.drop( ['BP'], axis = 1 )
# print(standardX)


# Read the verification file
test_df = pd.read_csv( '../04. 분석용 데이터/output_0929_26_2m.csv' )
print(test_df)
test_result_df = test_df

# DM =diabetes class
print('=======test_df======')
print(test_df.info())

# Cloumns Separation of  validation data
testX2 = test_df.drop( ['BP'], axis = 1 )
testy2 = test_df['BP']

print("==============================")


## Save the model to the computer - check the path
model = load_model('BloodPressure_high(ver1.3)_0930_01.h5')


# Source code that is actually applied
# realtest  predict
# Read and apply actual application data here!
# !!!!!!!!!!!do not change!.!!!!!!!!!!!
# StandardScaler standardization work
scaler = StandardScaler()
X_stand= scaler.fit_transform(standardX)


# Applying StandardScaler here!
testX2 = scaler.transform(testX2)
print('======testy_predict======')
print(testX2)
testy2_predict = model.predict(testX2 )
print('======testy_predict======')

# evaluate act model
print("============ evaluate result real  ==========")
print_evaluate(testy2, testy2_predict, train=True)

#======================================================
# Comparison of predicted and actual values
cnt_diff_0 = 0
cnt_diff_1 = 0
cnt_diff_2 = 0
cnt_diff_5 = 0
cnt_diff_10 = 0
index_ = []
prediction_list = []

testy2_predict = model.predict(testX2)
for i in range(len(testy2_predict)):
    label = testy2[i]
    prediction = math.floor(testy2_predict[i])
    prediction_list.append(prediction)
    diff_val = abs(prediction-label).round(2)
    #print('{:}. RealSugar| {:} | PredictSugar| {:} | 차이값 | {:}'.format(i, label, prediction, diff_val ))
    diff_val = abs(prediction-label).round(0)
    if diff_val > 10 :
        cnt_diff_10 += 1
        index_.append(i)
    elif diff_val > 5 :
        cnt_diff_5 += 1
    elif diff_val > 2 :
        cnt_diff_2 += 1
    elif diff_val > 0 :
        cnt_diff_1 += 1
    elif diff_val == 0 :
        cnt_diff_0 += 1
#======================================================
print('='*25)
print( f'검증 데이터 개수 : {len(testy2_predict)}' )
print( f'차이 없음 : {cnt_diff_0}' )
print('='*25)
print( f'0이상 차이나는 값들 : {cnt_diff_1}' )
print( f'2이상 차이나는 값들 : {cnt_diff_2}' )
print( f'5이상 차이나는 값들 : {cnt_diff_5}' )
print( f'10이상 차이나는 값들 : {cnt_diff_10}' )
#print( f'10차이 인덱스 : {index_}' )
print('총 데이터 결과의 평균 == RealBP| {:} | PredictBP| {:}'.format(label, sum(prediction_list)/len(prediction_list)))


# In[81]:


#=========================================
# import matplotlib.pyplot as plt


def clarke_error_grid(ref_values, pred_values, title_string):
    #Checking to see if the lengths of the reference and prediction arrays are the same
    assert (len(ref_values) == len(pred_values)), "Unequal number of values (reference : {}) (prediction : {}).". format(len(ref_values), len(pred_values))
    #Checks to see if the values are within the normal physiological range, otherwise it gives a warning
    if max(ref_values) > 400 or max(pred_values) > 400:
        print ("Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal physiological range of glucose (<400 mg/dl).".format(max(ref_values), max(pred_values)))
    if min(ref_values) < 0 or min(pred_values) < 0:
        print ("Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.".format(min(ref_values),  min(pred_values)))
    #Clear plot
    plt.clf()
    #Set up plot
    plt.scatter(ref_values, pred_values, marker='o', color='black', s=8)
    plt.title(title_string + " Clarke Error Grid")
    plt.xlabel("Reference Concentration (mg/dl)")
    plt.ylabel("Prediction Concentration (mg/dl)")
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.gca().set_facecolor('white')
    #Set axes lengths
    plt.gca().set_xlim([0, 400])
    plt.gca().set_ylim([0, 400])
    plt.gca().set_aspect((400)/(400))
    #Plot zone lines
    plt.plot([0,400], [0,400], ':', c='black')                      #Theoretical 45 regression line
    plt.plot([0, 175/3], [70, 70], '-', c='black')
    #plt.plot([175/3, 320], [70, 400], '-', c='black')
    plt.plot([175/3, 400/1.2], [70, 400], '-', c='black')           #Replace 320 with 400/1.2 because 100*(400 - 400/1.2)/(400/1.2) =  20% error
    plt.plot([70, 70], [84, 400],'-', c='black')
    plt.plot([0, 70], [180, 180], '-', c='black')
    plt.plot([70, 290],[180, 400],'-', c='black')
    # plt.plot([70, 70], [0, 175/3], '-', c='black')
    plt.plot([70, 70], [0, 56], '-', c='black')                     #Replace 175.3 with 56 because 100*abs(56-70)/70) = 20% error
    # plt.plot([70, 400],[175/3, 320],'-', c='black')
    plt.plot([70, 400], [56, 320],'-', c='black')
    plt.plot([180, 180], [0, 70], '-', c='black')
    plt.plot([180, 400], [70, 70], '-', c='black')
    plt.plot([240, 240], [70, 180],'-', c='black')
    plt.plot([240, 400], [180, 180], '-', c='black')
    plt.plot([130, 180], [0, 70], '-', c='black')
    #Add zone titles
    plt.text(30, 15, "A", fontsize=15)
    plt.text(370, 260, "B", fontsize=15)
    plt.text(280, 370, "B", fontsize=15)
    plt.text(160, 370, "C", fontsize=15)
    plt.text(160, 15, "C", fontsize=15)
    plt.text(30, 140, "D", fontsize=15)
    plt.text(370, 120, "D", fontsize=15)
    plt.text(30, 370, "E", fontsize=15)
    plt.text(370, 15, "E", fontsize=15)

    #Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        if (ref_values[i] <= 70 and pred_values[i] <= 70) or (pred_values[i] <= 1.2*ref_values[i] and pred_values[i] >= 0.8*ref_values[i]):
            zone[0] += 1    #Zone A
        elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (ref_values[i] <= 70 and pred_values[i] >= 180):
            zone[4] += 1    #Zone E
        elif ((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110) or ((ref_values[i] >= 130 and ref_values[i] <= 180) and (pred_values[i] <= (7/5)*ref_values[i] - 182)):
            zone[2] += 1    #Zone C
        elif (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180)) or (ref_values[i] <= 175/3 and pred_values[i] <= 180 and pred_values[i] >= 70) or ((ref_values[i] >= 175/3 and ref_values[i] <= 70) and pred_values[i] >= (6/5)*ref_values[i]):
            zone[3] += 1    #Zone D
        else:
            zone[1] += 1    #Zone B
    return plt, zone


# In[82]:


#=========================================
# function parameter
#  def clarke_error_grid(ref_values, pred_values, title_string): 
#========================================== 
print('=======clarke_error_grid region data ======')
plt, zone = clarke_error_grid(testy2, testy2_predict, 'Blood Pressure Validation')
print(zone)
plt.show()


# In[ ]:




