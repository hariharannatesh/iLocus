import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
import serial
import pickle
lis=[1,2,5,6,9,10,13,14]
lis1=[]
j=0
#ser=serial.Serial("COM9",115200)
df=pd.read_csv('Database90.txt')
#df=df.apply(pd.to_numeric)

data = np.array(df)
np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data);
y=np.array(data[:,0])
X=np.array(np.delete(data, [0], 1))
print(X)
print(y)
#X=np.array(df.drop(['Cellnumber'],1))
#y=np.array(df['Cellnumber'])


X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)
print(X_train,y_train)
clf=neighbors.KNeighborsClassifier(n_neighbors=9)
clf.fit(X_train,y_train)
data_pickle=open('trainned_data90.pkl','wb')
pickle.dump(clf,data_pickle)

accuracy=clf.score(X_test,y_test)
print(accuracy)
data_pickle.close()
