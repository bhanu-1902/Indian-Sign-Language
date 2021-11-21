import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sbn

data=pd.read_csv('D:/Indian Sign Language/Code/Code/Classification/csv files/train.csv',low_memory=False)
data.head()

test=pd.read_csv('D:/Indian Sign Language/Code/Code/Classification/csv files/test.csv',low_memory=False)
test.head()

x=data.iloc[:,0:150]
y=data.iloc[:,-1]

le=LabelEncoder()
y=le.fit_transform(y)

x_test=test.iloc[:,0:150]
y_test=test.iloc[:,-1]

y_test=le.fit_transform(y_test)

# Linear Kernel
cls=svm.SVC(kernel='linear')

cls.fit(x,y)

y_pred=cls.predict(x_test)

print(metrics.accuracy_score(y_test,y_pred))

print(classification_report(y_test,y_pred))

print(classification_report(y_test,y_pred)) #150

import pickle

file_name='D:/Indian Sign Language/Code/Code/Predict signs/files/SVM'
outfile=open(file_name,'wb')
pickle.dump(cls,outfile)
outfile.close()

c_m=confusion_matrix(y_test,y_pred)

plt.figure(figsize=(20,17))
plt.title("Confusion Matrix for K Nearest Neighbour")
df_cm=pd.DataFrame(c_m)
sbn.heatmap(df_cm,annot=True)

