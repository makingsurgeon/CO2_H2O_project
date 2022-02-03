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
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import seaborn as sns
#%%
#Remove data that have missing values
data = pd.read_excel("Solubility_database5-11.xlsx", header=1)
clean_data = data[data["Phases"] == "liq"]
clean_data1 = data[data["Phases"] == "liq+fl"]
clean_data2 = data[data["Phases"].isnull()]
data = pd.concat([clean_data, clean_data1, clean_data2], ignore_index = True)
#%%
data = data[data["xSiO2"].notna()]
data = data[data["PCO2 (bar)"].notna()]
data = data[data["CO2 Glass (wt.%)"].notna()]
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
#%%
idx = [9,11,30,47,122,162,163,164,165,166,167,168,169,170,171,0]
reduced_data = copy_of_data[:,idx]
#%%
for i in range(5,15):
    reduced_data[:,i] = reduced_data[:,i]/100
#%%
idx1 = []
for i in range(np.shape(reduced_data)[0]):
    if reduced_data[i,11]+reduced_data[i,12]+reduced_data[i,13] != 0:
        idx1.append(i)

reduced_data = reduced_data[idx1]
reduced_data = reduced_data[reduced_data[:,3]!=0]
#%%
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
reduced_data = reduced_data[reduced_data[:,4]!=0]
y_whole_set = np.log(reduced_data[:,4].astype("float"))
#%%
new_train = np.ones((np.shape(reduced_data)[0],8))
new_train[:,1] = reduced_data[:,7]/(reduced_data[:,11]+reduced_data[:,12]+reduced_data[:,13])
new_train[:,2] = (reduced_data[:,8]+reduced_data[:,10])
new_train[:,3] = (reduced_data[:,12]+reduced_data[:,13])
new_train[:,4] = np.log(reduced_data[:,3].astype("float"))
NBO = 2*(reduced_data[:,8]+reduced_data[:,10]+reduced_data[:,11]+reduced_data[:,12]+reduced_data[:,13]-reduced_data[:,7])
NBOO = NBO/(2*reduced_data[:,5]+2*reduced_data[:,6]+3*reduced_data[:,7]+reduced_data[:,8]+reduced_data[:,10]+reduced_data[:,11]+reduced_data[:,12]+reduced_data[:,13])
new_train[:,5] = NBOO

a  = reduced_data[:,0].astype("float")
b = reduced_data[:,1].astype("float")
new_train[:,6] = a/b
new_train[:,7] = 1/reduced_data[:,1]
#%%
new_train1 = np.append(new_train, np.reshape(reduced_data[:,15],(-1,1)),1)
new_train1 = np.append(new_train1, np.reshape(reduced_data[:,4],(-1,1)),1)
#%%
#Train-validation-test split: 60-20-20, using ordinary linear regression
X_train, X_test, y_train, y_test = train_test_split(new_train1, y_whole_set, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
#%% 
reg = LinearRegression().fit(X_train[:,:8], y_train)
beta = reg.coef_
beta[0] = reg.intercept_
y_val_pred = np.matmul(X_val[:,:8],beta)
val_error = np.sum((y_val_pred-y_val)**2)/len(y_val_pred)#1.2973  #validation error(MSE)

y_test_pred = np.matmul(X_test[:,:8],beta)
test_error = np.sum((y_test_pred-y_test)**2)/len(y_test_pred)#1.0566  #test error(MSE)
#%%
neigh = KNeighborsRegressor()
#%%
neigh.fit(X_train, y_train)
y_nn_predict_val = neigh.predict(X_val)
y_nn_predict_test = neigh.predict(X_test)
nn_val = np.sum((y_nn_predict_val-y_val)**2)/len(y_val) #1.3209
nn_test = np.sum((y_nn_predict_test-y_test)**2)/len(y_test) #0.665
#%%
svr = SVR(kernel = "linear", C=10)
#%%
svr.fit(X_train, y_train)
y_svm_predict_val = svr.predict(X_val)
y_svm_predict_test = svr.predict(X_test)
svm_val = np.sum((y_svm_predict_val-y_val)**2)/len(y_val) #1.8973
svm_test = np.sum((y_svm_predict_test-y_test)**2)/len(y_test) #1.0208

#%%
a = X_train[:,:8].astype("float64")
#%%
y_pred = np.zeros(len(y_train))
for i in range(np.shape(X_train)[0]):
    a = np.zeros(np.shape(X_train)[0])
    for j in range(np.shape(X_train)[0]):
        a[j] = np.linalg.norm(X_train[j,:8]-X_train[i,:8])
    b = a
    for k in range(len(b)):
        if b[k] == 0:
            continue
        else:
            b[k] = 1/b[k]
    wls_model = sm.WLS(y_train, X_train[:,:8].astype("float64"), weights = b)
    r = wls_model.fit()
    p = r.predict(exog = X_train[i,:8].astype("float64"))
    y_pred[i] = p
#%%
val_error_new = np.sum((y_pred-y_train)**2)/len(y_train) #1.0767 #train: 0.4917
#%%
y_pred_test = np.zeros(len(y_test))
for i in range(np.shape(X_test)[0]):
    a = np.zeros(np.shape(X_train)[0])
    for j in range(np.shape(X_train)[0]):
        a[j] = np.linalg.norm(X_train[j,:8]-X_test[i,:8])
    b = a
    for k in range(len(b)):
        if b[k] == 0:
            continue
        else:
            b[k] = 1/b[k]
    wls_model = sm.WLS(y_train, X_train[:,:8].astype("float64"), weights = b)
    r = wls_model.fit()
    p = r.predict(exog = X_test[i,:8].astype("float64"))
    y_pred_test[i] = p
test_error_new = np.sum((y_pred_test-y_test)**2)/len(y_test) #0.513
#%%
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
y_pred_test = np.exp(y_pred_test)/10000
#%%
plt.scatter(y_test, y_pred_test)
plt.xlabel("measured CO2")
plt.ylabel("calculated CO2")
#%%
clf = LocalOutlierFactor()
y_pred_test = np.reshape(y_pred_test,(-1,1))
new_y_pred = clf.fit_predict(y_pred_test) 
#%%
mask = new_y_pred != -1
#%%
booArray = y_test < 15
#%%
y_test_new = y_test[booArray]
#%%
y_test_pred_new = y_pred_test[booArray]
#%%
x = np.linspace(4, 13, 1000)
#%%
plt.scatter(y_test_new, y_test_pred_new)
plt.plot(x,x,"-k")
plt.xlabel("measured CO2 in wt%")
plt.ylabel("calculated CO2")
plt.title("test data")
#%%
y_train = np.exp(y_train)/10000
y_pred = np.exp(y_pred)/10000
#%%
y_train = np.delete(y_train, 126)
y_pred = np.delete(y_pred, 126)
#%%
df = np.append(np.reshape(y_train, (-1,1)), np.reshape(y_pred, (-1,1)),1)
#%%
a2 = X_train[:,8]
#%%
a2 = np.delete(a2,126)
df = np.append(df, np.reshape(a2, (-1,1)),1)
df = pd.DataFrame(df, columns = ["true_CO2", "calculated_CO2", "experiments"])
#%%
a1 = len(df["experiments"].unique())
#%%
marker = ['o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'H','o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'H','o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'H','o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'H','o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'H','o', 'x', '^', '+', '*', '8']
markers = [marker[i] for i in range(len(df["experiments"].unique()))]
sns.lmplot('true_CO2', 'calculated_CO2', data=df, hue='experiments', markers = markers, fit_reg=False)
#plt.plot(x,x,"-k")
plt.show()


#%%
y_test = np.delete(y_test, 123)
y_pred_test = np.delete(y_pred_test,123)
#%%
a3 = X_test[:,8]
#%%
a3 = np.delete(a3,123)
#%%
df1 = np.append(np.reshape(y_test, (-1,1)), np.reshape(y_pred_test, (-1,1)),1)
df1 = np.append(df1, np.reshape(a3, (-1,1)),1)
df1 = pd.DataFrame(df1, columns = ["true_value", "calculated_value", "experiments"])
#%%
c = len(df1["experiments"].unique())
#%%
marker1 = ['o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'H','o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'H','o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'H','o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'H','o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'H','o']
markers1 = [marker[i] for i in range(len(df1["experiments"].unique()))]
sns.lmplot('true_value', 'calculated_value', data=df1, hue='experiments', markers = markers1, fit_reg=False)
plt.show()

#%%
a4 = X_train[:,9]
a4 = np.delete(a4, 126)
#%%
df = np.append(np.reshape(np.log(a4.astype("float64")), (-1,1)), np.reshape(y_pred, (-1,1)),1)
df = np.append(df, np.reshape(a2, (-1,1)),1)
df = pd.DataFrame(df, columns = ["log_PCO2", "calculated_CO2", "experiments"])
sns.lmplot('log_PCO2', 'calculated_CO2', data=df, hue='experiments', fit_reg=False)
plt.show()
#%%
df = np.append(np.reshape(a4, (-1,1)), np.reshape(y_pred, (-1,1)),1)
df = np.append(df, np.reshape(a2, (-1,1)),1)
df = pd.DataFrame(df, columns = ["PCO2", "calculated_CO2", "experiments"])
sns.lmplot('PCO2', 'calculated_CO2', data=df, hue='experiments', fit_reg=False)
plt.show()

