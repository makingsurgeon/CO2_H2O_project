#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:48:24 2022

@author: zihuiouyang
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import seaborn as sns
#%%
#Remove data that have missing values
data = pd.read_excel("Solubility_database5-12.xlsx", header=1)
clean_data = data[data["Phases"] == "liq"]
clean_data1 = data[data["Phases"] == "liq+fl"]
clean_data2 = data[data["Phases"].isnull()]
data = pd.concat([clean_data, clean_data1, clean_data2], ignore_index = True)
data = data[data["PCO2 (bar)"].notna()]
data = data[data["CO2 Glass (wt.%)"].notna()]
data = data[data["Reference"]!= "Eguchi, J., Dasgupta, R. (2017)"]
data = data[data["Reference"]!= "Eguchi, J., Dasgupta, R. (2018)"]
data = data.to_numpy()
copy_of_data = data
#%%
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
index2 = []
for i in range(np.shape(copy_of_data)[0]):
    if copy_of_data[i,122] != 0:
        index2.append(i)
copy_of_data = copy_of_data[index2]
for i in range(np.shape(copy_of_data)[0]):
    if np.isnan(copy_of_data[i,179]):
        copy_of_data[i,179] = 0
for i in range(np.shape(copy_of_data)[0]):
    if np.isnan(copy_of_data[i,180]):
        copy_of_data[i,180] = 0
for i in range(np.shape(copy_of_data)[0]):
    if np.isnan(copy_of_data[i,181]):
        copy_of_data[i,181] = 0
for i in range(np.shape(copy_of_data)[0]):
    if np.isnan(copy_of_data[i,182]):
        copy_of_data[i,182] = 0
for i in range(np.shape(copy_of_data)[0]):
    if np.isnan(copy_of_data[i,183]):
        copy_of_data[i,183] = 0
for i in range(np.shape(copy_of_data)[0]):
    if np.isnan(copy_of_data[i,184]):
        copy_of_data[i,184] = 0
for i in range(np.shape(copy_of_data)[0]):
    if np.isnan(copy_of_data[i,185]):
        copy_of_data[i,185] = 0
for i in range(np.shape(copy_of_data)[0]):
    if np.isnan(copy_of_data[i,186]):
        copy_of_data[i,186] = 0
for i in range(np.shape(copy_of_data)[0]):
    if np.isnan(copy_of_data[i,187]):
        copy_of_data[i,187] = 0
for i in range(np.shape(copy_of_data)[0]):
    if np.isnan(copy_of_data[i,188]):
        copy_of_data[i,188] = 0
#%%
idx = [9,11,189,47,122,179,180,181,182,183,184,185,186,187,188,0]
reduced_data = copy_of_data[:,idx]
for i in range(5,15):
    reduced_data[:,i] = reduced_data[:,i]/100
reduced_data[:,2] = reduced_data[:,2]/100
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
for i in range(np.shape(reduced_data)[0]):
    for j in range(15):
        if np.isnan(reduced_data[i,j]):
            reduced_data[i,j] = 0
#%%
reduced_data = reduced_data[reduced_data[:,4]!=0]
y_whole_set = np.log(reduced_data[:,4].astype("float"))
new_train = np.ones((np.shape(reduced_data)[0],11))
new_train[:,1] = (reduced_data[:,5]+reduced_data[:,7])**2+1e-7
new_train[:,2] = reduced_data[:,6]+reduced_data[:,8]
new_train[:,3] = reduced_data[:,10]
new_train[:,4] = reduced_data[:,11]
new_train[:,5] = reduced_data[:,12]
new_train[:,6] = reduced_data[:,13]
new_train[:,7] = 1/reduced_data[:,1]
new_train[:,8] = reduced_data[:,2]
new_train[:,9] = (reduced_data[:,5]+reduced_data[:,7])+1e-5
new_train[:,10] = (reduced_data[:,5]+reduced_data[:,7])**3+1e-9
#%%
X_train, X_test, y_train, y_test = train_test_split(new_train, y_whole_set, test_size=0.2, random_state=5)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
#%%
for i in range(np.shape(X_train)[0]):
    for j in range(np.shape(X_train)[1]):
        if np.isnan(X_train[i,j]):
            X_train[i,j] = 0
for i in range(np.shape(X_val)[0]):
    for j in range(np.shape(X_val)[1]):
        if np.isnan(X_val[i,j]):
            X_val[i,j] = 0
for i in range(np.shape(X_test)[0]):
    for j in range(np.shape(X_test)[1]):
        if np.isnan(X_test[i,j]):
            X_test[i,j] = 0

#%%
y_pred1 = np.zeros(len(y_val))
for i in range(np.shape(X_val)[0]):
    a = np.zeros(np.shape(X_train)[0])
    li = []
    for l in range(np.shape(X_val)[1]):
        if X_val[3,l] != 0:
            li.append(l)
    for j in range(np.shape(X_train)[0]):
        a[j] = np.linalg.norm(X_train[j,li]-X_val[i,li])
    b = a
    for k in range(len(b)):
        if b[k] == 0:
            continue
        else:
            b[k] = 1/b[k]
    wls_model = sm.WLS(y_train, X_train.astype("float64")[:,li], weights = b)
    r = wls_model.fit()
    p = r.predict(exog = X_val[i,li].astype("float64"))
    y_pred1[i] = p
val_error_new = np.sum((y_pred1-y_val)**2)/len(y_val) #1.0829 0.7365  0.8681 1.2639  1.006 Avg:0.9915
# 1.1543 1.5042 3.8406 0.866 0.9138 1.0859
#%%
y_val = np.exp(y_val)/10000
y_pred1 = np.exp(y_pred1)/10000
#%%
plt.scatter(y_val,y_pred1)
plt.xlabel("measured CO2")
plt.ylabel("calculated CO2")
#%%
a = (1.0829+0.7365+0.8681+1.2639+1.006)/5
