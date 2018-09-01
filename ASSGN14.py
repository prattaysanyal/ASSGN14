from sklearn import model_selection
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
%matplotlib inline
df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data")
df
df.head()
df.shape
X=df.drop(["B"],axis=1)
Y=df["B"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)
from sklearn import tree
model=tree.DecisionTreeClassifier(max_leaf_nodes=10)
model.fit(X_train,Y_train)
pred = model.predict(X_test)
from sklearn import metrics
metrics.accuracy_score(Y_test,pred)
from sklearn.metrics import classification_report
print(classification_report(Y_test,pred))
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,pred)
model1=tree.DecisionTreeClassifier(criterion="entropy",max_depth=10)
model1.fit(X_train,Y_train)
pred1 = model1.predict(X_test)
metrics.accuracy_score(Y_test,pred1)
print(classification_report(Y_test,pred1))
confusion_matrix(Y_test,pred1)
from sklearn.ensemble import RandomForestClassifier
model2=RandomForestClassifier(random_state=1,max_depth=10)
model2.fit(X_train,Y_train)
pred2 = model2.predict(X_test)
metrics.accuracy_score(Y_test,pred2)
print(classification_report(Y_test,pred2))
confusion_matrix(Y_test,pred2)
model3=RandomForestClassifier(criterion="entropy")
model3.fit(X_train,Y_train)
pred3 = model3.predict(X_test)
metrics.accuracy_score(Y_test,pred3)
print(classification_report(Y_test,pred3))
confusion_matrix(Y_test,pred3)

