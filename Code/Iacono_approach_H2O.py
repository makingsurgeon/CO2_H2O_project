#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 20:34:16 2021

@author: Bihong
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import statsmodels.api as sm
#%%
#Remove data that have missing values
data = pd.read_excel("Solubility_database5-11.xlsx", header=1)
data = data[data["xSiO2"].notna()]
data = data[data["PH2O (bar)"].notna()]
data = data[data["H2O Glass (wt.%)"].notna()]
data = data.to_numpy()
copy_of_data = data
#%%

copy_of_data = copy_of_data[copy_of_data[:,46]!=0]
#Data used for fitting the model
idx = [9,11,46,118,162,163,164,165,166,167,168,169,170,171]
reduced_data = copy_of_data[:,idx]

idx1 = []
for i in range(np.shape(reduced_data)[0]):
    if (2*reduced_data[i,4]+2*reduced_data[i,5]+3*reduced_data[i,6]+reduced_data[i,7]+reduced_data[i,9]+reduced_data[i,10]+reduced_data[i,11]+reduced_data[i,12]) != 0:
        idx1.append(i)

reduced_data = reduced_data[idx1]
reduced_data = reduced_data[reduced_data[:,3]!=0]

idx2 = []
for i in range(np.shape(reduced_data)[0]):
    if type(reduced_data[i,0]) is str:
        idx2.append(i)
reduced_data = np.delete(reduced_data, idx2, 0)

idx3 = []
for i in range(np.shape(reduced_data)[0]):
    if type(reduced_data[i,1]) is str:
        idx3.append(i)
reduced_data = np.delete(reduced_data, idx3, 0)
#%%
idx1 = []
for i in range(np.shape(reduced_data)[0]):
    if reduced_data[i,10]+reduced_data[i,11]+reduced_data[i,12] != 0:
        idx1.append(i)

reduced_data = reduced_data[idx1]

#%%
new_train = np.ones((np.shape(reduced_data)[0],8))
new_train[:,1] = reduced_data[:,6]/(reduced_data[:,10]+reduced_data[:,11]+reduced_data[:,12])
new_train[:,2] = (reduced_data[:,7]+reduced_data[:,9])
new_train[:,3] = (reduced_data[:,11]+reduced_data[:,12])
new_train[:,4] = np.log(reduced_data[:,2].astype("float"))
NBO = 2*(reduced_data[:,7]+reduced_data[:,9]+reduced_data[:,10]+reduced_data[:,11]+reduced_data[:,12]-reduced_data[:,6])
NBOO = NBO/(2*reduced_data[:,4]+2*reduced_data[:,5]+3*reduced_data[:,6]+reduced_data[:,7]+reduced_data[:,9]+reduced_data[:,10]+reduced_data[:,11]+reduced_data[:,12])
new_train[:,5] = NBOO

a  = reduced_data[:,0].astype("float")
b = reduced_data[:,1].astype("float")
new_train[:,6] = a/b
new_train[:,7] = 1/reduced_data[:,1]

y_whole_set = np.log(reduced_data[:,3].astype("float"))
#%%
#Train-validation-test split: 60-20-20, using ordinary linear regression
X_train, X_test, y_train, y_test = train_test_split(new_train, y_whole_set, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 

reg = LinearRegression().fit(X_train, y_train)
beta = reg.coef_
beta[0] = reg.intercept_

y_val_pred = np.matmul(X_val,beta)
val_error = np.sum((y_val_pred-y_val)**2)/len(y_val_pred)#0.293

y_test_pred = np.matmul(X_test,beta)
test_error = np.sum((y_test_pred-y_test)**2)/len(y_val_pred) #0.3433
#%%
y_pred = np.zeros(len(y_val))
for i in range(np.shape(X_val)[0]):
    a = np.zeros(np.shape(X_train)[0])
    for j in range(np.shape(X_train)[0]):
        a[j] = np.linalg.norm(X_train[j]-X_val[i])
    b = a
    for k in range(len(b)):
        if b[k] == 0:
            continue
        else:
            b[k] = 1/b[k]
    wls_model = sm.WLS(y_train, X_train, weights = b)
    r = wls_model.fit()
    p = r.predict(exog = X_val[i])
    y_pred[i] = p
#%%
val_error_new = np.sum((y_pred-y_val)**2)/len(y_val) #0.1358
#%%
y_pred_test = np.zeros(len(y_test))
for i in range(np.shape(X_test)[0]):
    a = np.zeros(np.shape(X_train)[0])
    for j in range(np.shape(X_train)[0]):
        a[j] = np.linalg.norm(X_train[j]-X_test[i])
    b = a
    for k in range(len(b)):
        if b[k] == 0:
            continue
        else:
            b[k] = 1/b[k]
    wls_model = sm.WLS(y_train, X_train, weights = b)
    r = wls_model.fit()
    p = r.predict(exog = X_test[i])
    y_pred_test[i] = p
test_error_new = np.sum((y_pred_test-y_test)**2)/len(y_test) #0.1863
#%%
rid = Ridge(alpha=100).fit(X_train, y_train)
beta_r = rid.coef_
beta_r[0] = rid.intercept_
y_val_pred_r = np.matmul(X_val,beta_r)
val_error_r = np.sum((y_val_pred_r-y_val)**2)/len(y_val_pred)#0.2022  #validation error(MSE)
y_test_pred_r = np.matmul(X_test,beta_r)
test_error_r = np.sum((y_test_pred_r-y_test)**2)/len(y_val_pred)#0.2721  #test error(MSE)

las = Lasso(alpha=0.00001).fit(X_train, y_train)
beta_l = las.coef_
beta_l[0] = las.intercept_
y_val_pred_l = np.matmul(X_val,beta_l)
val_error_l = np.sum((y_val_pred_l-y_val)**2)/len(y_val_pred)#0.1238 #validation error(MSE)
y_test_pred_l = np.matmul(X_test,beta_l)
test_error_l = np.sum((y_test_pred_l-y_test)**2)/len(y_val_pred)#0.2739  #test error(MSE)
#%%
y_test = np.exp(y_test)
y_test_pred = np.exp(y_test_pred)
#%%
plt.scatter(y_test, y_test_pred)
plt.xlabel("measured H2O")
plt.ylabel("calculated H2O")
#%%
y_pred = np.zeros(len(y_test))
for i in range(np.shape(X_test)[0]):
    a = np.zeros(np.shape(X_train)[0])
    for j in range(np.shape(X_train)[0]):
        a[j] = np.linalg.norm(X_train[j]-X_test[i])
    b = a
    for k in range(len(b)):
        if b[k] == 0:
            continue
        else:
            b[k] = 1/b[k]
    wls_model = sm.WLS(y_train, X_train, weights = b)
    r = wls_model.fit()
    p = r.predict(exog = X_test[i])
    y_pred[i] = p
test_error_new = np.sum((y_pred-y_test)**2)/len(y_test) #0.5898
#%%
booArray = y_test < 10
#%%
y_test_new = y_test[booArray]
#%%
y_test_pred_new = y_test_pred[booArray]
#%%
plt.scatter(y_test_new, y_test_pred_new)
x = np.linspace(0, 10, 1000)
plt.plot(x,x,"-k")
plt.xlabel("measured H2O in wt%")
plt.ylabel("calculated H2O")
plt.title("test data")


