# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:03:06 2022

@author: mahon
"""

import pandas as pd
import sklearn
import numpy as np
import operator as op


data = pd.read_csv(r'C:\Users\mahon\Documents\Python Scripts\NN Practice\Decision_data.csv',sep= ',')

#Split into features and outcome variables. Note Y is categorical
X=data.values[:,1:4]
Y=data.values[:,0]

#Split into test and training sets
X_train, X_test, Y_train,Y_test=sklearn.model_selection.train_test_split(X,Y,test_size=.3)

#%% Lets calculate the information gain from splitting X
#Auxilary Functions



def entropy_single(a):

    n=len(set(a.flatten()))
    counter=np.zeros(n)
    
    for i in range(1,n+1):
        print(i)
        counter[i-1]=op.countOf(a, i)
    
    counter=counter/sum(counter)
    
    ent=0
    
    for i in range(n):
        ent+=-(counter[i]*np.log(counter[i])/np.log(2))
    
    return ent


def entropy_double(test):
        
    na=len(set(test[:,0]))
    nb=len(set(test[:,1]))
    
    counter=np.zeros([na,nb])
    
    for i in range(na):
        
        sub=test[test[:,0]==(i+1),:]
        count_vec=sub[:,1]
        
        for j in range(nb):
            ct=op.countOf(count_vec, (j+1))
            counter[i,j]=ct
    
        size=sum(sum(counter))
        ent=0
    
        for i in range(na):
            outter=counter[i,:]/sum(counter[i,:])
            inner=-np.log(outter)/np.log(2)
            ent+=(sum(counter[i,:])/size)*np.dot(inner,outter)
    
    return ent

#Information gain is the difference
IG=entropy_single(X[:,2:3])-entropy_double(X[:,1:3])

#Now lets calculate which binary partition will yield the highest IG

entropy_single(X[:,0])-entropy_double(X[:,0:2])
entropy_single(X[:,1])-entropy_double(X[:,1:3])
entropy_single(X[:,2])-entropy_double(X[:,1:3])

entropy_single(X[:,0])-entropy_double(np.transpose(np.array([X[:,0],X[:,2]])))
entropy_single(X[:,1])-entropy_double(np.transpose(np.array([X[:,0],X[:,1]])))
entropy_single(X[:,2])-entropy_double(np.transpose(np.array([X[:,0],X[:,2]])))







