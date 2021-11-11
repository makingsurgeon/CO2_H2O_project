#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 18:26:33 2021

@author: Bihong
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
#%%
#Remove data that have missing values
data = pd.read_excel("Solubility_database5-10.xlsx", header=1)
data = data[data["SiO2 (wt.% dry)"].notna()]
data = data[data["PCO2 (bar)"].notna()]
data = data[data["CO2 Glass (wt.%)"].notna()]
data = data.to_numpy()
copy_of_data = data

for i in range(np.shape(copy_of_data)[0]):
    if np.isnan(data[i,30]) and not np.isnan(data[i,26]):
        copy_of_data[i,30] = copy_of_data[i,26]
index = []
for i in range(np.shape(copy_of_data)[0]):
    if not np.isnan(copy_of_data[i,30]):
        index.append(i)
copy_of_data = copy_of_data[index]

for i in range(np.shape(copy_of_data)[0]):
    if np.isnan(data[i,122]) and not np.isnan(data[i,120]):
        copy_of_data[i,122] = copy_of_data[i,120]*10000
index1 = []
for i in range(np.shape(copy_of_data)[0]):
    if not np.isnan(copy_of_data[i,122]):
        index1.append(i)
copy_of_data = copy_of_data[index1]
#Data used for fitting the model
copy_of_data = copy_of_data[copy_of_data[:,120]!=0]
idx = [9,11,30,47,122,127,129,131,133,135,137,139,141,143,145]
reduced_data = copy_of_data[:,idx]

idx1 = []
for i in range(np.shape(reduced_data)[0]):
    if reduced_data[i,11]+reduced_data[i,12]+reduced_data[i,13] != 0:
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

y_whole_set = np.log(reduced_data[:,4].astype("float"))
new_train = np.ones((np.shape(reduced_data)[0],8))
new_train[:,1] = reduced_data[:,2]
new_train[:,2] = reduced_data[:,7]/(reduced_data[:,11]+reduced_data[:,12]+reduced_data[:,13])
new_train[:,3] = (reduced_data[:,8]+reduced_data[:,10])/100
new_train[:,4] = (reduced_data[:,12]+reduced_data[:,13])/100
new_train[:,5] = np.log(reduced_data[:,3].astype("float"))
NBO = 2*(reduced_data[:,8]+reduced_data[:,10]+reduced_data[:,11]+reduced_data[:,12]+reduced_data[:,13]-reduced_data[:,7])
NBOO = NBO/(2*reduced_data[:,5]+2*reduced_data[:,6]+3*reduced_data[:,7]+reduced_data[:,8]+reduced_data[:,10]+reduced_data[:,11]+reduced_data[:,12]+reduced_data[:,13])
new_train[:,6] = NBOO

a  = reduced_data[:,0].astype("float")
b = reduced_data[:,1].astype("float")
new_train[:,7] = a/b
#Train-validation-test split: 60-20-20, using ordinary linear regression
X_train, X_test, y_train, y_test = train_test_split(new_train, y_whole_set, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 
reg = LinearRegression().fit(X_train, y_train)
beta = reg.coef_
beta[0] = reg.intercept_
y_val_pred = np.matmul(X_val,beta)
val_error = np.sum((y_val_pred-y_val)**2)/len(y_val_pred)#1.5526  #validation error(MSE)

y_test_pred = np.matmul(X_test,beta)
test_error = np.sum((y_test_pred-y_test)**2)/len(y_val_pred)#0.966  #test error(MSE)

#Ridge Regression
rid = Ridge(alpha=0.1).fit(X_train, y_train)
beta_r = rid.coef_
beta_r[0] = rid.intercept_
y_val_pred_r = np.matmul(X_val,beta_r)
val_error_r = np.sum((y_val_pred_r-y_val)**2)/len(y_val_pred)#1.56  #validation error(MSE)
y_test_pred_r = np.matmul(X_test,beta_r)
test_error_r = np.sum((y_test_pred_r-y_test)**2)/len(y_val_pred)#0.9648  #test error(MSE)
#LASSO
las = Lasso(alpha=0.0001).fit(X_train, y_train)
beta_l = las.coef_
beta_l[0] = las.intercept_
y_val_pred_l = np.matmul(X_val,beta_l)
val_error_l = np.sum((y_val_pred_l-y_val)**2)/len(y_val_pred)#1.5523  #validation error(MSE)
y_test_pred_l = np.matmul(X_test,beta_l)
test_error_l = np.sum((y_test_pred_l-y_test)**2)/len(y_val_pred)#0.966  #test error(MSE)
#%%
y_test = np.exp(y_test)/10000
y_test_pred = np.exp(y_test_pred)/10000
#%%
plt.scatter(y_test, y_test_pred)
plt.xlabel("measured CO2")
plt.ylabel("calculated CO2")
#%%
clf = LocalOutlierFactor()
y_test_pred = np.reshape(y_test_pred,(-1,1))
new_y_pred = clf.fit_predict(y_test_pred) 
#%%
mask = new_y_pred != -1
#%%
y_test_new = y_test[mask]
#%%
y_test_pred_new = y_test_pred[mask,:]
#%%
plt.scatter(y_test_new, y_test_pred_new)
plt.xlabel("measured CO2")
plt.ylabel("calculated CO2")
#%%






