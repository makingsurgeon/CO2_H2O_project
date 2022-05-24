#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:30:21 2022

@author: zihuiouyang
"""

import argparse
import numpy as np
import statsmodels.api as sm
import pandas as pd
import os


parser = argparse.ArgumentParser(description='Predicting CO2 percentages')
parser.add_argument("data", metavar="DIR",
                    help='path to dataset in accordance with the template')
parser.add_argument("-o","--output", metavar="DIR", default = "", 
                    help="location of the output csv file (default current directory)")


def H2O_prediction(args):
    data = pd.read_excel("Solubility_database5-12.xlsx", header=1)
    clean_data = data[data["Phases"] == "liq"]
    clean_data1 = data[data["Phases"] == "liq+fl"]
    clean_data2 = data[data["Phases"].isnull()]
    
    data = pd.concat([clean_data, clean_data1, clean_data2], ignore_index = True)
    data = data[data["PH2O (bar)"].notna()]
    data = data[data["H2O Glass (wt.%)"].notna()]
    data = data[data["Reference"]!= "Eggler, D.H., Rosenhauer, M. (1978)"]
    data = data[data["Reference"]!= "Hui, H., Zhang, Y., Xu, Z., Behrens, H. (2008)"]
    data = data[data["Reference"]!= "Hui, H., Zhang, Y., Xu, Z., Behrens, H. (2008)"]
    data = data[data["Reference"]!= "Muth, M., Duncan, M., Dasgupta, R. (2020)"]
    data = data[data["Reference"]!= "Duncan, M. S., Dasgupta, R. (2014)"]
    data = data[data["Reference"]!= "Duncan, M. S., Dasgupta, R. (2015)"]
    data = data[data["Reference"]!= "Silver, L., Stolper, E. (1989)"]
    data = data[data["Reference"]!= "King, P.L., Holloway, J.R. (2002)"]
    data = data.to_numpy()
    copy_of_data = data
    
    copy_of_data = copy_of_data[copy_of_data[:,46]!=0]
    #Data used for fitting the model
    idx = [9,11,46,179,180,181,182,183,184,185,186,187,188,118]
    reduced_data = copy_of_data[:,idx]


    idx1 = []
    for i in range(np.shape(reduced_data)[0]):
        if (2*reduced_data[i,3]+2*reduced_data[i,4]+3*reduced_data[i,5]+reduced_data[i,6]+reduced_data[i,8]+reduced_data[i,9]+reduced_data[i,10]+reduced_data[i,11]) > 0:
            idx1.append(i)

    reduced_data = reduced_data[idx1]
    reduced_data = reduced_data[reduced_data[:,2]!=0]

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
    
    idx1 = []
    for i in range(np.shape(reduced_data)[0]):
        if reduced_data[i,9]+reduced_data[i,10]+reduced_data[i,11] != 0:
            idx1.append(i)

    reduced_data = reduced_data[idx1]
    
    new_train = np.ones((np.shape(reduced_data)[0],5))
    new_train[:,1] = np.log(reduced_data[:,2].astype("float"))
    NBO = 2*(reduced_data[:,6]+reduced_data[:,8]+reduced_data[:,9]+reduced_data[:,10]+reduced_data[:,11]-reduced_data[:,5])
    NBOO = NBO/(2*reduced_data[:,3]+2*reduced_data[:,4]+3*reduced_data[:,5]+reduced_data[:,6]+reduced_data[:,8]+reduced_data[:,9]+reduced_data[:,10]+reduced_data[:,11])
    new_train[:,2] = NBOO
    

    a  = reduced_data[:,0].astype("float")
    b = reduced_data[:,1].astype("float")
    new_train[:,3] = a/b
    new_train[:,4] = 1/reduced_data[:,1]

    new_train1 = new_train
    
    y_whole_set = np.log(reduced_data[:,13].astype("float"))
    
    y1 = pd.read_csv(args.data)
    selected_rows = y1[~y1['Name'].isnull()]
    cols = selected_rows.columns.tolist()
    selected_rows = selected_rows[selected_rows["CO2 (ppm)"].isnull()]
    new_selected_rows = [cols[14],cols[16],cols[15]]+cols[1:11]+[cols[0]]
    reduced_data2 = selected_rows[new_selected_rows]
    reduced_data2 = reduced_data2.to_numpy()
    
    new_train2 = np.ones((np.shape(reduced_data2)[0],5))
    new_train2[:,1] = np.log(reduced_data2[:,2].astype("float")*reduced_data2[:,0].astype("float")*10)
    NBO1 = 2*(reduced_data2[:,6]+reduced_data2[:,8]+reduced_data2[:,9]+reduced_data2[:,10]+reduced_data2[:,11]-reduced_data2[:,5])
    NBOO1 = NBO1/(2*reduced_data2[:,3]+2*reduced_data2[:,4]+3*reduced_data2[:,5]+reduced_data2[:,6]+reduced_data2[:,8]+reduced_data2[:,9]+reduced_data2[:,10]+reduced_data2[:,11])
    new_train2[:,2] = NBOO1
    a1  = reduced_data2[:,0].astype("float")
    b1 = reduced_data2[:,1].astype("float")
    new_train2[:,3] = a1/b1
    new_train2[:,4] = 1/reduced_data2[:,1]

    y_pred = np.zeros(np.shape(new_train2)[0])
    
    for i in range(len(y_pred)):
        a2 = np.zeros(np.shape(new_train1)[0])
        for j in range(np.shape(new_train1)[0]):
            a2[j] = np.linalg.norm(new_train1[j]-new_train2[i])
        b2 = a2
        for k in range(len(b2)):
            if b2[k] == 0:
                continue
            else:
                b2[k] = 1/b2[k]
        wls_model = sm.WLS(y_whole_set, new_train.astype("float64"), weights = b)
        r = wls_model.fit()
        p = r.predict(exog = new_train2[i].astype("float64"))
        y_pred[i] = p
    
    y_pred = np.exp(y_pred)
    names = reduced_data2[:,13]
    df = pd.DataFrame({'Names': names, 'H2O %': y_pred})
    
    return df
                      
def CO2_prediction(args):
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
        if np.isnan(data[i,179]):
            copy_of_data[i,179] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,180]):
            copy_of_data[i,180] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,181]):
            copy_of_data[i,181] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,182]):
            copy_of_data[i,182] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,183]):
            copy_of_data[i,183] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,184]):
            copy_of_data[i,184] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,185]):
            copy_of_data[i,185] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,186]):
            copy_of_data[i,186] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,187]):
            copy_of_data[i,187] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,188]):
            copy_of_data[i,188] = 0
    
    idx = [9,11,47,122,179,180,181,182,183,184,185,186,187,188]
    reduced_data = copy_of_data[:,idx]
    
    idx1 = []
    for i in range(np.shape(reduced_data)[0]):
        if reduced_data[i,10]+reduced_data[i,11]+reduced_data[i,12] > 0:
            idx1.append(i)

    reduced_data = reduced_data[idx1]
    reduced_data = reduced_data[reduced_data[:,3]!=0]
    reduced_data = reduced_data[reduced_data[:,2]!=0]
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
    
    
    idx4 = []
    for i in range(np.shape(reduced_data)[0]):
        if reduced_data[i,4] + reduced_data[i,6] < 0.35:
            idx4.append(i)
    
    reduced_data = np.delete(reduced_data, idx4, 0)
    new_train = np.ones((np.shape(reduced_data)[0],8))
    new_train[:,1] = reduced_data[:,6]/(reduced_data[:,10]+reduced_data[:,11]+reduced_data[:,12])
    new_train[:,2] = (reduced_data[:,7]+reduced_data[:,9])
    new_train[:,3] = (reduced_data[:,11]+reduced_data[:,12])
    new_train[:,4] = np.log(reduced_data[:,2].astype("float"))
    NBO = 2*(reduced_data[:,8]+reduced_data[:,10]+reduced_data[:,11]+reduced_data[:,12]+reduced_data[:,13]-reduced_data[:,7])
    NBOO = NBO/(2*reduced_data[:,5]+2*reduced_data[:,6]+3*reduced_data[:,7]+reduced_data[:,8]+reduced_data[:,10]+reduced_data[:,11]+reduced_data[:,12]+reduced_data[:,13])
    new_train[:,5] = NBOO
    
    

    a  = reduced_data[:,0].astype("float")
    b = reduced_data[:,1].astype("float")
    new_train[:,6] = a/b
    new_train[:,7] = 1/reduced_data[:,1]
    
    new_train1 = new_train

    y_whole_set = np.log(reduced_data[:,3].astype("float"))
    
    y1 = pd.read_csv(args.data)
    selected_rows = y1[~y1['Name'].isnull()]
    selected_rows = selected_rows[selected_rows["CO2 (ppm)"].isnull()]
    cols = selected_rows.columns.tolist()
    new_selected_rows = [cols[0],cols[14],cols[16],cols[15]]+cols[1:11]
    reduced_data2 = selected_rows[new_selected_rows]
    reduced_data2 = reduced_data2.to_numpy()
    
    
    new_train2 = np.ones((np.shape(reduced_data2)[0],8))
    new_train2[:,1] = reduced_data2[:,6]/(reduced_data2[:,10]+reduced_data2[:,11]+reduced_data2[:,12])
    new_train2[:,2] = (reduced_data2[:,7]+reduced_data2[:,9])
    new_train2[:,3] = (reduced_data2[:,11]+reduced_data2[:,12])
    new_train2[:,4] = np.log(reduced_data2[:,3].astype("float")*10)
    NBO1 = 2*(reduced_data2[:,8]+reduced_data2[:,10]+reduced_data2[:,11]+reduced_data2[:,12]+reduced_data2[:,13]-reduced_data2[:,7])
    NBOO1 = NBO1/(2*reduced_data2[:,5]+2*reduced_data2[:,6]+3*reduced_data2[:,7]+reduced_data2[:,8]+reduced_data2[:,10]+reduced_data2[:,11]+reduced_data2[:,12]+reduced_data2[:,13])
    new_train2[:,5] = NBOO1

    a1  = reduced_data2[:,1].astype("float")
    b1 = reduced_data2[:,2].astype("float")
    new_train2[:,6] = a1/b1
    new_train2[:,7] = 1/reduced_data2[:,2]
    

    y_pred = np.zeros(np.shape(new_train2)[0])
    for i in range(len(y_pred)):
        a2 = np.zeros(np.shape(new_train1)[0])
        for j in range(np.shape(new_train1)[0]):
            a2[j] = np.linalg.norm(new_train1[j]-new_train2[i])
        b2 = a2
        for k in range(len(b2)):
            if b2[k] == 0:
                continue
            else:
                b2[k] = 1/b2[k]
        wls_model = sm.WLS(y_whole_set, new_train.astype("float64"), weights = b)
        r = wls_model.fit()
        p = r.predict(exog = new_train2[i].astype("float64"))
        y_pred[i] = p
     
    out = np.exp(y_pred)
    names = reduced_data2[:,0]
    df = pd.DataFrame({'Names': names, 'CO2 %': out})
    return df
    
def Pressure(args):
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
        if np.isnan(data[i,179]):
            copy_of_data[i,179] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,180]):
            copy_of_data[i,180] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,181]):
            copy_of_data[i,181] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,182]):
            copy_of_data[i,182] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,183]):
            copy_of_data[i,183] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,184]):
            copy_of_data[i,184] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,185]):
            copy_of_data[i,185] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,186]):
            copy_of_data[i,186] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,187]):
            copy_of_data[i,187] = 0
    for i in range(np.shape(copy_of_data)[0]):
        if np.isnan(data[i,188]):
            copy_of_data[i,188] = 0
    
    idx = [9,11,47,122,179,180,181,182,183,184,185,186,187,188]
    reduced_data = copy_of_data[:,idx]
    
    idx1 = []
    for i in range(np.shape(reduced_data)[0]):
        if reduced_data[i,10]+reduced_data[i,11]+reduced_data[i,12] > 0:
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
    idx4 = []
    for i in range(np.shape(reduced_data)[0]):
        if reduced_data[i,4] + reduced_data[i,6] < 0.35:
            idx4.append(i)
    
    reduced_data = np.delete(reduced_data, idx4, 0)
    
    new_train = np.ones((np.shape(reduced_data)[0],6))
    new_train[:,1] = reduced_data[:,6]/(reduced_data[:,10]+reduced_data[:,11]+reduced_data[:,12])
    new_train[:,2] = (reduced_data[:,7]+reduced_data[:,9])
    new_train[:,3] = (reduced_data[:,11]+reduced_data[:,12])
    new_train[:,4] = np.log(reduced_data[:,3].astype("float"))
    NBO = 2*(reduced_data[:,8]+reduced_data[:,10]+reduced_data[:,11]+reduced_data[:,12]+reduced_data[:,13]-reduced_data[:,7])
    NBOO = NBO/(2*reduced_data[:,5]+2*reduced_data[:,6]+3*reduced_data[:,7]+reduced_data[:,9]+reduced_data[:,10]+reduced_data[:,11]+reduced_data[:,12]+reduced_data[:,13])
    new_train[:,5] = NBOO

    
    new_train1 = new_train
    
    y_whole_set = np.log(reduced_data[:,0].astype("float"))
    
    y1 = pd.read_csv(args.data)
    selected_rows = y1[~y1['Name'].isnull()]
    cols = selected_rows.columns.tolist()
    selected_rows = selected_rows[~selected_rows["CO2 (ppm)"].isnull()]
    new_selected_rows = [cols[0],cols[12],cols[13]]+cols[1:11]
    reduced_data2 = selected_rows[new_selected_rows]
    reduced_data2 = reduced_data2.to_numpy()
    
    new_train2 = np.ones((np.shape(reduced_data2)[0],6))
    new_train2[:,1] = reduced_data2[:,5]/(reduced_data2[:,9]+reduced_data2[:,10]+reduced_data2[:,11])
    new_train2[:,2] = (reduced_data2[:,6]+reduced_data2[:,8])
    new_train2[:,3] = (reduced_data2[:,10]+reduced_data2[:,11])
    new_train2[:,4] = np.log(reduced_data2[:,2].astype("float")*10)
    NBO1 = 2*(reduced_data2[:,7]+reduced_data2[:,9]+reduced_data2[:,10]+reduced_data2[:,11]+reduced_data2[:,12]-reduced_data2[:,6])
    NBOO1 = NBO1/(2*reduced_data2[:,4]+2*reduced_data2[:,5]+3*reduced_data2[:,6]+reduced_data2[:,7]+reduced_data2[:,9]+reduced_data2[:,10]+reduced_data2[:,11]+reduced_data2[:,12])
    new_train2[:,5] = NBOO1


    
    

    y_pred = np.zeros(np.shape(new_train2)[0])
    for i in range(len(y_pred)):
        a2 = np.zeros(np.shape(new_train1)[0])
        for j in range(np.shape(new_train1)[0]):
            a2[j] = np.linalg.norm(new_train1[j]-new_train2[i])
        b2 = a2
        for k in range(len(b2)):
            if b2[k] == 0:
                continue
            else:
                b2[k] = 1/b2[k]
        wls_model = sm.WLS(y_whole_set, new_train1.astype("float64"), weights = b2)
        r = wls_model.fit()
        p = r.predict(exog = new_train2[i].astype("float64"))
        y_pred[i] = p
     
    out = np.exp(y_pred)
    
    names = reduced_data2[:,0]
    df = pd.DataFrame({'Names': names, 'Pressure': out})
    
    return df

def main():
    args = parser.parse_args()
    H2O_pred = H2O_prediction(args)
    CO2_pred = CO2_prediction(args)
    CO2_pres = Pressure(args)
    H2O_pred["CO2 %"] = CO2_pred["CO2 %"]
    if args.output == "" :
        H2O_pred.to_csv("percentage.csv")
        CO2_pres.to_csv("pressure.csv")
    else:
        p1 = os.path.join(args.output,"percentage.csv")
        p2 = os.path.join(args.output,"pressure.csv")
        H2O_pred.to_csv(p1)
        CO2_pres.to_csv(p2)
    
if __name__ == '__main__':
    main()
    
