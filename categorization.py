# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 12:08:38 2022

@author: mahon
"""

#####Sk learn to impliment SVM and K-means clustering
#####We're just going to use the sklearn library here
#####In other scripts I'll recreate the algorithim from scratch

####Lets start by loading the libraries we'll need, the we'll build some fake data

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


 
# Next, we create datasets X containing n_samples -- each X has 
# a corresponding Y containing two classes -- 0 and 1
size=200

X,Y = make_blobs(n_samples=size, centers=2,
                  random_state=0, cluster_std=0.40)

#Make Y into a 2d array -- this will prove convenient later on
Y=Y[:,np.newaxis]
data=pd.DataFrame(data=np.concatenate([X,Y],axis=1),column=['x1','x2','y'])

# Plot fake data
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring');
plt.show() 

#Now lets split out data into testing and training samples
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1337)

#These data are highly bifurcated so k-means and SVM should word well here
#Lets start with the SVM since our data already have categorical labels

#Import the svm library from sklearn
from sklearn import svm

#Itialize the classifier -- linear sets the classifier to use a linear hyperplane
sv_classifier=sklearn.svm.SVC(kernel='linear')

#Run the SVM fit -- note that we convert y_train back to a 1-dim array
y_train=y_train.flatten()
sv_classifier.fit(X_train,y_train)

#Lets conduct some predictions and see how well our model does
y_pred = sv_classifier.predict(X_test)

#Calculate errors
y_test=y_test.flatten()
total_errors=np.sum(np.abs(y_test-y_pred))

#No errors! Not suprising given how bifurcated our data are. 
#Lets prace using sklearns built in evaluation procedures

from sklearn import metrics

# Model Accuracy is percent of correct predictions -- 100% again
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Okay now lets do some clustering 
from sklearn import cluster

centers, labels, obj_value=sklearn.cluster.k_means(X, n_clusters=2)

total_errors=np.sum(np.abs(Y.flatten()-labels))















