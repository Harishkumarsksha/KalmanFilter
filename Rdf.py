import os 
import sys 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

class BattRdf:
    
    def __init__(self,Size=0.3):
        self.data = None 
        self.Rdf_train  =None
        self.SOC_test   =None
        self.SOC_train  =None
        self.Rdf_pred   =None

        self.SOC = None
        self.OCV =None
        self.interceptK = 0
        self.Rdf_coef = 1
        self.model = LinearRegression()
        # self.model = linear_model.Ridge(alpha=.5)
        self.testSize = Size
        self.loadData()
        self.sliptData()
        self.modelFit()
        self.modelPredict()

    def loadData(self):
        self.data = pd.read_csv('BMS_lookup_tables/Table_RTTC2_SOC_v2.txt', sep=",",header=None,names=["SOC","Rdf"],dtype="float")
        self.SOC = self.data["SOC"].values
        self.Rdf = self.data["Rdf"].values
        self.SOC = np.asarray(self.SOC)
        self.SOC=self.SOC.reshape(len(self.SOC),1)
        self.Rdf = np.asarray(self.Rdf)
        self.Rdf=self.Rdf.reshape(len(self.Rdf),1)


    def sliptData(self):
        self.SOC_train, self.SOC_test, self.Rdf_train, self.Rdf_pred = train_test_split(self.SOC, self.Rdf, test_size=self.testSize, random_state=0)
        # self.OCV_train.reshape(len(self.OCV_train),1)
        # print(f'{len(self.OCV_train)}')
    
    def modelFit(self):
        self.model.fit(self.SOC_train, self.Rdf_train)
        self.interceptK=self.model.intercept_
        self.Rdf_coef=self.model.coef_

        print(f'Model intercept : {self.interceptK} and  model Coeffient : {self.Rdf_coef}')
    
    def modelPredict(self):
        self.Rdf_pred = self.model.predict(self.SOC_test)
        print(f'model predicted output for {self.SOC_test[10]} : {self.model.predict(self.SOC_test[10].reshape(1,1))}')

    def RdfFunc(self,SOC):
        return self.Rdf_coef * 99.09 + self.interceptK


if __name__ == "__main__":
    
    K=BattRdf()
    print(K.RdfFunc(50.1))
    
    
