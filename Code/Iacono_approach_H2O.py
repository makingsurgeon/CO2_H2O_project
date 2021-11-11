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
#%%
#Remove data that have missing values
data = pd.read_excel("Solubility_database5-10.xlsx", header=1)
data = data[data["SiO2 (wt.% dry)"].notna()]
data = data[data["PH2O (bar)"].notna()]
data = data[data["H2O Glass (wt.%)"].notna()]
data = data.to_numpy()
copy_of_data = data

for i in range(np.shape(copy_of_data)[0]):
    if np.isnan(data[i,122]) and not np.isnan(data[i,120]):
        copy_of_data[i,122] = copy_of_data[i,120]*10000
index1 = []
for i in range(np.shape(copy_of_data)[0]):
    if not np.isnan(copy_of_data[i,122]):
        index1.append(i)
copy_of_data = copy_of_data[index1]

copy_of_data = copy_of_data[copy_of_data[:,46]!=0]
#Data used for fitting the model
idx = [9,11,46,118,127,129,131,133,135,137,139,141,143,145]
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

y_whole_set = np.log(reduced_data[:,3].astype("float")*10000)

new_train = np.ones((np.shape(reduced_data)[0],4))
new_train[:,1] = np.log(reduced_data[:,2].astype("float"))

NBO = 2*(reduced_data[:,7]+reduced_data[:,9]+reduced_data[:,10]+reduced_data[:,11]+reduced_data[:,12]-reduced_data[:,6])
NBOO = NBO/(2*reduced_data[:,4]+2*reduced_data[:,5]+3*reduced_data[:,6]+reduced_data[:,7]+reduced_data[:,9]+reduced_data[:,10]+reduced_data[:,11]+reduced_data[:,12])
new_train[:,2] = NBOO

a  = reduced_data[:,0].astype("float")
b = reduced_data[:,1].astype("float")
new_train[:,3] = a/b
#Train-validation-test split: 60-20-20, using ordinary linear regression
X_train, X_test, y_train, y_test = train_test_split(new_train, y_whole_set, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 

reg = LinearRegression().fit(X_train, y_train)
beta = reg.coef_
beta[0] = reg.intercept_

y_val_pred = np.matmul(X_val,beta)
val_error = np.sum((y_val_pred-y_val)**2)/len(y_val_pred)#0.1238

y_test_pred = np.matmul(X_test,beta)
test_error = np.sum((y_test_pred-y_test)**2)/len(y_val_pred) #0.2739

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
y_test = np.exp(y_test)/10000
y_test_pred = np.exp(y_test_pred)/10000
#%%
plt.scatter(y_test, y_test_pred)
plt.xlabel("measured H2O")
plt.ylabel("calculated H2O")
