import os 
import sys 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

class BattCdf:
    
    def __init__(self,Size=0.3):
        self.data = None 
        self.Cdf_train  =None
        self.SOC_test   =None
        self.SOC_train  =None
        self.Cdf_pred   =None

        self.SOC = None
        self.OCV =None
        self.interceptK = 0
        self.Cdf_coef = 1
        self.model = LinearRegression()
        # self.model = linear_model.Ridge(alpha=.5)
        self.testSize = Size
        self.loadData()
        self.sliptData()
        self.modelFit()
        self.modelPredict()

    def loadData(self):
        self.data = pd.read_csv('BMS_lookup_tables/Table_CTTC2_SOC_v2.txt', sep=",",header=None,names=["SOC","Cdf"],dtype="float")
        self.SOC = self.data["SOC"].values
        self.Cdf = self.data["Cdf"].values
        self.SOC = np.asarray(self.SOC)
        self.SOC=self.SOC.reshape(len(self.SOC),1)
        self.Cdf = np.asarray(self.Cdf)
        self.Cdf=self.Cdf.reshape(len(self.Cdf),1)


    def sliptData(self):
        self.SOC_train, self.SOC_test, self.Cdf_train, self.Cdf_pred = train_test_split(self.SOC, self.Cdf, test_size=self.testSize, random_state=0)
        # self.OCV_train.reshape(len(self.OCV_train),1)
        # print(f'{len(self.OCV_train)}')
    
    def modelFit(self):
        self.model.fit(self.SOC_train, self.Cdf_train)
        self.interceptK=self.model.intercept_
        self.Cdf_coef=self.model.coef_

        print(f'Model intercept : {self.interceptK} and  model Coeffient : {self.Cdf_coef}')
    
    def modelPredict(self):
        self.Cdf_pred = self.model.predict(self.SOC_test)
        print(f'model predicted output for {self.SOC_test[10]} : {self.model.predict(self.SOC_test[10].reshape(1,1))}')

    def CdfFunc(self,SOC):
        return self.Cdf_coef * 99.09 + self.interceptK


# if __name__ == "__main__":
    
#     K=BattCdf()
#     print(K.CdfFunc(99.09722))
    
    
