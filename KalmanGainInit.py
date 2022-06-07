import os 
import sys 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

class KalmanGainInit:
    
    def __init__(self,Size=0.3):
        self.data = None 
        self.OCV_train  =None
        self.OCV_test   =None
        self.SOC_train  =None
        self.SOC_pred   =None

        self.SOC = None
        self.OCV =None
        self.interceptK = 0
        self.Kalman_coef = 1
        self.model = LinearRegression()
        # self.model = linear_model.Ridge(alpha=.5)
        self.testSize = Size
        self.loadData()
        self.sliptData()
        self.modelFit()
        self.modelPredict()
        self.plot()

    def loadData(self):
        self.data = pd.read_csv('BMS_lookup_tables/Table_OCV_SOC_2.txt', sep=",",header=None,names=["SOC","OCV"],dtype="float")
        self.SOC = self.data["SOC"].values
        self.OCV = self.data["OCV"].values
        self.SOC = np.asarray(self.SOC)
        self.SOC=self.SOC.reshape(len(self.SOC),1)
        self.OCV = np.asarray(self.OCV)
        self.OCV=self.OCV.reshape(len(self.OCV),1)


    def sliptData(self):
        self.OCV_train, self.OCV_test, self.SOC_train, self.SOC_pred = train_test_split(self.OCV, self.SOC, test_size=self.testSize, random_state=0)
        # self.OCV_train.reshape(len(self.OCV_train),1)
        # print(f'{len(self.OCV_train)}')
    
    def modelFit(self):
        self.model.fit(self.OCV_train, self.SOC_train)
        self.interceptK=self.model.intercept_
        self.Kalman_coef=self.model.coef_

        print(f'Model Kalman intercept : {self.interceptK} and  model Kalman Gain init : {self.Kalman_coef}')
    
    def modelPredict(self):
        self.SOC_pred = self.model.predict(self.OCV_test)
        print(f'model predicted output for {self.OCV_test[10]} : {self.model.predict(self.OCV_test[10].reshape(1,1))}')

    def SOC_0(self,OCV):
        # print(f'model predicted output for {OCV} : {self.Kalman_coef * OCV + self.interceptK}')
        return self.Kalman_coef * OCV + self.interceptK

    def plot(self):
        # f, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.plot(self.OCV,self.SOC,'b-')
        # ax1.set_title("SOC and Open circuit voltage ")
        # ax2.plot(self.OCV_test,self.SOC_pred,'r.')
        
        # plt.plot(self.OCV,self.SOC,'b-')
        # plt.plot(self.OCV_test,self.SOC_pred,'r.')
        # plt.show()
        pass

# if __name__ == "__main__":
    
#     K=KalmanGainInit()
#     K.SOC_0(4.0497944)
    
