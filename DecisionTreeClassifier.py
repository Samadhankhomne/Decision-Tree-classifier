# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset  = pd.read_csv("Social_Network_Ads.csv")

x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:,-1].values

# Spliting the dataset into the Training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y ,test_size=0.20, random_state=0)

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.fit_transform(X_test)
'''

# Training the Decision tree classification model on the training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy",max_depth=4)
classifier.fit(X_train,y_train)

'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=2,n_estimators=30,criterion="entropy", random_state=1)
classifier.fit(X_train,y_train)
'''

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Making  the Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(X_train,y_train)
print("bias:",bias)

variance = classifier.score(X_test, y_test)
print("variance:",variance)


import os
os.getcwd()




