#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:37:31 2021

@author: zihuiouyang
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import statsmodels.api as sm
#%%
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
#%%
temperature = reduced_data[:,1]+273.15
d = (647.14/(0.2212**2)*6.93054*10**(-7))-(8.38293*10**(-8)/(0.2212**2)*temperature)
c = (647.14/(0.2212**1.5)*(-3.30558*10**(-5)))+(2.30524*10**(-6)/(0.2212**1.5)*temperature)
b = 9.18301*10**(-4)*647.14/0.2212
a = (647.14**2.5/(0.2212)*(5.45963*10**(-5)))-(8.6392*10**(-6)*647.14**1.5/(0.2212)*temperature)
p = reduced_data[:,0]/100
temperature = temperature.astype(np.float)
d = d.astype(np.float)
c = c.astype(np.float)
a = a.astype(np.float)
p = p.astype(np.float)
f = 8.3145*temperature*np.log(1000*p)+b*p
f = f+(a/(b*np.sqrt(temperature)))*(np.log(8.3145*temperature+b*p)-np.log(8.3145*temperature+2*b*p))
f = f+(2/3)*c*p**(3/2)+d/2*p**2
f = f/(8.3145*temperature)
f = np.exp(f)
reduced_data[:,2] = f
#%%
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
val_error = np.sum((y_val_pred-y_val)**2)/len(y_val_pred)#0.3952

y_test_pred = np.matmul(X_test,beta)
test_error = np.sum((y_test_pred-y_test)**2)/len(y_val_pred) #0.4223
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
val_error_new = np.sum((y_pred-y_val)**2)/len(y_val) #0.4288
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
test_error_new = np.sum((y_pred_test-y_test)**2)/len(y_test) #0.5051




