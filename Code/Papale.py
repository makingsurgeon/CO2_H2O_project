#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 11:19:03 2022

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
#%%
data = pd.read_excel("Solubility_database5-10.xlsx", header=1)
clean_data = data[data["Phases"] == "liq"]
clean_data1 = data[data["Phases"] == "liq+fl"]
clean_data2 = data[data["Phases"].isnull()]
#%%
new_data = pd.concat([clean_data, clean_data1, clean_data2], ignore_index = True)
data = new_data.to_numpy()
copy_of_data = data
#%%
for i in range(np.shape(copy_of_data)[0]):
    if np.isnan(data[i,122]) and not np.isnan(data[i,120]):
        copy_of_data[i,122] = copy_of_data[i,120]*10000
index1 = []
for i in range(np.shape(copy_of_data)[0]):
    if not np.isnan(copy_of_data[i,122]):
        index1.append(i)
copy_of_data = copy_of_data[index1]
#%%
#Data used for fitting the model
copy_of_data = copy_of_data[copy_of_data[:,120]!=0]
#%%
idx = [9,11,30,47,122,127,129,131,133,135,137,139,141,143,145]
#%%
reduced_data = copy_of_data[:,idx]
#%%

