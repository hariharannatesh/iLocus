import numpy as np
from sklearn import preprocessing,cross_validation,neighbors
import pandas as pd
import pickle
import serial
import time
ser=serial.Serial("COM9",115200)

count=0
#data=np.array(data)
#data=np.reshape(1,-1)
data_pickle=open('trainned_data90.pkl','rb')
clf=pickle.load(data_pickle)
while 1:
    m=ser.readline()
    m=str(m)
    m=m[2:]
    m=m[:15]
    sum1=int(m[1])*(-10)+int(m[2])*(-1)
    sum2=int(m[5])*(-10)+int(m[6])*(-1)
    sum3=int(m[9])*(-10)+int(m[10])*(-1)
    sum4=int(m[13])*(-10)+int(m[14])*(-1)
    lis1=[sum1,sum2,sum3,sum4]
    lis1=np.array(lis1)
    lis1=lis1.reshape(1,-1)
    prediction=clf.predict(lis1)
    count=count+1
    #x,y=coordinates(prediction)
    #print(prediction)
    prediction=prediction.astype(np.int8)
    print(prediction)
    
    ser.write(prediction)
    time.sleep(0.5)



data_pickle.close()
