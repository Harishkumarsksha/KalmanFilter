from turtle import shape
import numpy as np
import pandas as pd 
import time
import matplotlib.pyplot as plt
from scipy import rand 

from KalmanGainInit import KalmanGainInit
from Cdf import BattCdf
from Rdf import BattRdf
from Ri import BattRi

class KalmanFilter:
    

    def __init__(self,del_t=0.1):
        
        
        self.OCV = self.voltageGenerate()
        self.Kinit=KalmanGainInit()
        self.SOC0  = self.Kinit.SOC_0(self.OCV)*10
        self.SOC   = self.SOC0
        
        self.KGain = self.Kinit.Kalman_coef
        self.Kalman_intercept = self.Kinit.interceptK   
        self.Vdf = 0
        
        self.Current = 0
        self.Cnom   = 53


        self.vk = np.random.uniform(-0.001,0.001)# process noise varaince 
        self.wk = np.random.uniform(-0.002,0.002) #measurement noise varaince 
    #     noise = (1/np.sqrt(2*np.pi*np.square(self.vk)))*np.exp((0.1**2/2*self.vk**2))
        self.X_priori = np.ones(2) # Priori estimate Xk/k-1 state [SOC,Vdf] nx1
        self.X_update = np.ones(2) # Priori estimate Xk/k-1 state [SOC,Vdf] nx1
    #     self.P_priori = np.identity(len(self.X_priori)) *0.001 # Priori estimate Pk/k-1 covaraince of the priori state estimate nxn 
        
        self.P_priori=np.ones([2,2]) 
        self.P_update=np.ones([2,2]) 
        self.Fx   =None # state Transitin matrix (SOC,Vdf) nxn matrix 
        self.Fu   =None # Input Control matrix (Current Control) nx1
        self.Q    =np.identity(len(self.X_priori)) * self.vk# np.round(np.random.uniform(self.vk-0.00001,self.vk),2) # process noise Covaraince              nxn
        self.R    =np.identity(len(self.X_priori)) * self.wk # Measurement noise covariance matrix    nxn

        self.H    =None # observation odel which maps the state space  into the observed space mxn


        self.del_t = del_t



        self.BattCdf = BattCdf()   
        self.BattRdf = BattRdf()                    
        self.BattRi  = BattRi()
        
        self.parameterUpdate()
        self.X_update = np.asarray([self.SOC,self.Vdf],dtype=np.float64)

    def parameterUpdate(self):
        
        
        self.Cdf = self.BattCdf.CdfFunc(self.SOC)
        self.Rdf = self.BattRdf.RdfFunc(self.SOC)
        self.Ri  = self.BattRi.RiFunc(self.SOC)
        

      
    def voltageGenerate(self,min=3.6,max=4):
        return np.round(np.random.uniform(min,max),3)
    
    def currentGenerate(self,min=2.0,max=5.0):
        return np.round(np.random.uniform(min,max),3)
    

  
    def prioriStateEstiate_prediction(self):
        delta = (self.del_t/(self.Cdf*self.Rdf))
        Fx = np.array([[1,0],[0,delta]])
        Fu = np.asarray([(self.del_t/self.Cnom),(self.del_t/self.Cdf)])
        Fu = Fu.reshape(len(Fu),1)
        print(Fu.shape)
        
        
        
        self.X_priori = Fx * self.X_update + Fu*self.Current
        
        self.SOC_pred = np.array([self.KGain,1]) * self.X_update .T + self.Ri*self.Current
        
        print(f'SOC kalan prediction {self.SOC_pred}')
        
        self.P_priori = Fx * self.P_update * Fx.T + self.R
        
        
    
    def prioriUpdatePrediction(self):
        Cd = np.array([self.KGain,1])
        Cd = Cd.reshape(len(Cd),1)
        print( Cd.shape)
        a=self.P_priori * Cd
        b=Cd.T * a
        print(b)
        mk = np.linalg.inv( b  )
        
        self.KGain = self.P_priori * Cd.T * mk
        
        zk = self.SOC - self.SOC_pred
        self.X_update = self.X_priori + self.KGain * zk
        
        self.P_update = (np.eye(len(self.X_priori)) - self.KGain * Cd) * self.P_priori
        
        self.Current =  self.currentGenerate()
        self.columbCount()
        
        print(f'Kalan Gain uodate {self.KGain}')
        
    def columbCount(self):
        
        self.SOC = self.SOC0 + self.del_t*self.Current/(self.Cnom)
        print(f'Columb count SOC {self.SOC}')
        self.parameterUpdate()
    
if __name__ == "__main__":
    KF= KalmanFilter()
    
    for i in range (1,2):
        
        KF.prioriStateEstiate_prediction()
        KF.prioriUpdatePrediction()
        time.sleep(0.1)