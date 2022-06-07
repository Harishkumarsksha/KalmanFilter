import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from KalmanGainInit import KalmanGainInit
from Cdf import BattCdf
from Rdf import BattRdf
from Ri import BattRi

class KalmanFilter:

    def __init__(self):
        self.OCV = self.voltageGenerate()
        self.Kinit=KalmanGainInit()
        
        self.KGain = self.Kinit.Kalman_coef
        self.Kalman_intercept = self.Kinit.interceptK

        self.Vdf = 0
        
        
        self.BattCdf = BattCdf()   
        self.BattRdf = BattRdf()                    
        self.BattRi  = BattRi()



        self.Current = 0
        self.Cnom = 53 # battery capacity 

        self.parameterUpdate()

    def parameterUpdate(self):
        self.SOC = self.Kinit.SOC_0(self.OCV)*10
        self.X = np.array([self.SOC,self.Vdf])
        self.Cdf = self.BattCdf.CdfFunc(self.SOC)
        self.Rdf = self.BattRdf.RdfFunc(self.SOC)
        self.Ri  = self.BattRi.RiFunc(self.SOC)

        self.Fx=self.FxFunc()
        self.Fu=self.FuFunc()

        

    def stateUpdate(self):
        new_X = np.dot(self.Fx,self.X) + self.Fu * self.Current

        return new_X

    def FxFunc(self):
        delta = 1-(0.1/(self.Cdf*self.Rdf))
        return np.array([[1,0],[0,delta]])

    def FuFunc(self):
        return np.array([0.1/self.Cnom,0.1/(self.Cdf)])*self.Current



    def voltageGenerate(self,min=3.6,max=4):
        return np.round(np.random.uniform(min,max),3)

    def currentGenerate(self,min=2.0,max=5.0):
        return np.round(np.random.uniform(min,max),3)

    def vBattEstimate(self):
        return self.KGain*self.SOC + self.Ri*self.Current + self.Kalman_intercept

if __name__ == "__main__":
    KF= KalmanFilter()
    print(KF.OCV)
    print(KF.vBattEstimate())
    print(KF.stateUpdate())