#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:20:46 2021

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
#%%
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
#%%
temperature = reduced_data[:,1]+273.15
d = (304.2/(0.0738**2)*6.93054*10**(-7))-(8.38293*10**(-8)/(0.0738**2)*temperature)
c = (304.2/(0.0738**1.5)*(-3.30558*10**(-5)))+(2.30524*10**(-6)/(0.0738**1.5)*temperature)
b = 9.18301*10**(-4)*304.2/0.0738
a = (304.2**2.5/(0.0738)*(5.45963*10**(-5)))-(8.6392*10**(-6)*304.2**1.5/(0.0738)*temperature)
p = reduced_data[:,3]/1000
temperature = temperature.astype(np.float)
d = d.astype(np.float)
c = c.astype(np.float)
a = a.astype(np.float)
p = p.astype(np.float)
#%%
f = 8.3145*temperature*np.log(1000*p)+b*p
f = f+(a/(b*np.sqrt(temperature)))*(np.log(8.3145*temperature+b*p)-np.log(8.3145*temperature+2*b*p))
f = f+(2/3)*c*p**(3/2)+d/2*p**2
f = f/(8.3145*temperature)
f = np.exp(f)
f = f/reduced_data[:,3]
reduced_data[:,3] = f
#%%
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
val_error = np.sum((y_val_pred-y_val)**2)/len(y_val_pred)#1.1989  #validation error(MSE)

y_test_pred = np.matmul(X_test,beta)
test_error = np.sum((y_test_pred-y_test)**2)/len(y_test_pred)#0.8955  #test error(MSE)
#%%
y_pred = np.zeros(len(y_val))
for i in range(np.shape(X_val)[0]):
    a = np.zeros(np.shape(X_train)[0])
    for j in range(np.shape(X_train)[0]):
        a[j] = np.linalg.norm(X_train[j]-X_val[i], ord = 1)
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
val_error_new = np.sum((y_pred-y_val)**2)/len(y_val) #0.9392
#%%
y_pred_test = np.zeros(len(y_test))
for i in range(np.shape(X_test)[0]):
    a = np.zeros(np.shape(X_train)[0])
    for j in range(np.shape(X_train)[0]):
        a[j] = np.linalg.norm(X_train[j]-X_test[i], ord = 1)
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
test_error_new = np.sum((y_pred_test-y_test)**2)/len(y_test) #0.7657




