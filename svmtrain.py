import numpy as np
from sklearn import preprocessing, model_selection, neighbors,svm
import pandas as pd

df=pd.read_csv('Database100.txt')
#df.drop(['id'],1,inplace=True)

data=np.array(df)

acctotal=0

for i in range(10):
	
	np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data);
	y=np.array(data[:,0])
	X=np.array(np.delete(data, [0], 1))

#X=np.array(df.drop(['class'],1))
#y=np.array(df['class'])

	X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.3)
	clf=svm.SVC()
	clf.fit(X_train,y_train)

	accuracy=clf.score(X_test,y_test)
	print(accuracy)

	acctotal=acctotal+accuracy

print("Mean accuracy:",(acctotal)/10.0)