#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 20:02:22 2021

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
data = data[data["fCO2 (bar)"].notna()]
data = data[data["CO2 Glass (wt.%)"].notna()]
data = data.to_numpy()
copy_of_data = data
#%%
X = copy_of_data[:,[50,120]]
#%%
lof = LocalOutlierFactor()
yhat = lof.fit_predict(X)
#%%
plt.scatter(X[:,0], X[:,1])
#%%
mask = yhat != -1
#%%
X_t = X[mask,:]
#%%
plt.scatter(X_t[:,0], X_t[:,1])
#%%
data = pd.read_excel("Solubility_database5-10.xlsx", header=1)
data = data[data["CO2 Glass (wt.%)"].notna()]
data = data[data["H2O Glass (wt.%)"].notna()]
data = data.to_numpy()
copy_of_data = data
#%%
X = copy_of_data[:,[118,120]]
#%%
lof = LocalOutlierFactor()
yhat = lof.fit_predict(X)
#%%
plt.scatter(X[:,0], X[:,1])
#%%
mask = yhat != -1
#%%
X_t = X[mask,:]
#%%
plt.scatter(X_t[:,0], X_t[:,1])
