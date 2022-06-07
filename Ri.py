import os 
import sys 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

class BattRi:
    
    def __init__(self,Size=0.3):
        self.data = None 
        self.Ri_train  =None
        self.SOC_test   =None
        self.SOC_train  =None
        self.Ri_pred   =None

        self.SOC = None
        self.OCV =None
        self.interceptK = 0
        self.Ri_coef = 1
        self.model = LinearRegression()
        # self.model = linear_model.Ridge(alpha=.5)
        self.testSize = Size
        self.loadData()
        self.sliptData()
        self.modelFit()
        self.modelPredict()

    def loadData(self):
        self.data = pd.read_csv('BMS_lookup_tables/Table_Rin_SOC_charge.txt', sep=" ",header=None,names=["NAN1","SOC","NAN2","Ri","NAN3"])
        self.data.drop(["NAN1","NAN2","NAN3"], axis=1, inplace=True)
        # print(self.data["Ri"])
        self.SOC = self.data["SOC"].values
        self.Ri = self.data["Ri"].values
        self.SOC = np.asarray(self.SOC)
        self.SOC=self.SOC.reshape(len(self.SOC),1)
        self.Ri = np.asarray(self.Ri)
        self.Ri=self.Ri.reshape(len(self.Ri),1)


    def sliptData(self):
        self.SOC_train, self.SOC_test, self.Ri_train, self.Ri_pred = train_test_split(self.SOC, self.Ri, test_size=self.testSize, random_state=0)
        # self.OCV_train.reshape(len(self.OCV_train),1)
        # print(f'{len(self.OCV_train)}')
    
    def modelFit(self):
        self.model.fit(self.SOC_train, self.Ri_train)
        self.interceptK=self.model.intercept_
        self.Ri_coef=self.model.coef_

        print(f'Model intercept : {self.interceptK} and  model Coeffient : {self.Ri_coef}')
    
    def modelPredict(self):
        self.Ri_pred = self.model.predict(self.SOC_test)
        print(f'model predicted output for {self.SOC_test[10]} : {self.model.predict(self.SOC_test[10].reshape(1,1))}')

    def RiFunc(self,SOC):
        return self.Ri_coef * 99.09 + self.interceptK


# if __name__ == "__main__":
    
#     K=BattRi()
#     print(K.RiFunc(50.1))
    
    
