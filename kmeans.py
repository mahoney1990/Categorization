# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 12:53:27 2022

@author: mahon
"""
#K means from scratch

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

###First, define our distance function
def dist(x1,x2):
    inner=0
    n=len(x1)
    
    #Check lengths match
    if len(x1)!=len(x2):
        print("Get it together! Arrays must be of the same dimension!")
    
    #Build inner
    for i in range(n):
        inner+=(x1[i]-x2[i])**2
    
    #Square root that shit
    dist=(inner)**(1/2)
    return dist

#Now lets get some preliminary objects created -- generate some blobs of data, we'll cluster on X
size=200
X,Y = make_blobs(n_samples=size, centers=2,
                  random_state=0, cluster_std=1.2)

# Plot fake data
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring');
plt.show() 


n_clusters=2
ranges=np.zeros([2,2])

initial_means=np.zeros([n_clusters,2])
new_means=np.zeros([n_clusters,2])
old_means=np.zeros([n_clusters,2])

for i in range(n_clusters):
    ranges[i,0]=min(X[:,i])
    ranges[i,1]=max(X[:,i])

#Import random package for RNG
import random

#Generate random inital guesses for cluster means
for i in range(n_clusters):
    initial_means[i,0]=random.uniform(ranges[0,0],ranges[0,1])
    initial_means[i,1]=random.uniform(ranges[1,0],ranges[1,1])

#Define matrix of distances row = observation, col=cluster, entry = distance from cluster mean
mean_distances=np.zeros([size,n_clusters])

#Vector of indicators for closest cluster mean
cluster_label=np.zeros([size])

for j in range(n_clusters):
    #Grab cluster means
    x1=initial_means[j,:]
    
    #Loops through each X value, calculate distance to each mean
    for k in range(size):
        x=X[k,:]
        mean_distances[k,j]=dist(x,x1)

#Calculate category labels
for k in range(size):
    cluster_label[k]=np.argmin(mean_distances[k,:])

cluster_label=cluster_label[:,np.newaxis]

num_cat=np.zeros([n_clusters])
for i in range(n_clusters):
   num_cat[i]=(cluster_label==i).sum()

new_means[0,:]=sum(X*(1-cluster_label))/num_cat[0]
new_means[1,:]=sum(X*cluster_label)/num_cat[1]

#Now iterate until convergence

d=1
tol=.01
its=0

olds=np.zeros([n_clusters,2])
olds=new_means

while d>tol:
    its+=1
    print(its)
    
    for j in range(n_clusters):
        #Grab cluster means
        x1=olds[j,:]
        
        #Loops through each X value, calculate distance to each mean
        for k in range(size):
            x=X[k,:]
            mean_distances[k,j]=dist(x,x1)
    
    #Calculate category labels
    for k in range(size):
        cluster_label[k]=np.argmin(mean_distances[k,:])
        
    for k in range(n_clusters):
         num_cat[k]=(cluster_label==k).sum()
    
    #new_means[0,:]=sum(X*(1-cluster_label))/num_cat[0]
    #new_means[1,:]=sum(X*cluster_label)/num_cat[1]
    
    new1=sum(X*(1-cluster_label))/num_cat[0]
    new2=sum(X*cluster_label)/num_cat[1]

    new_means=np.vstack([new1,new2])

    d=max(dist(new_means[0,:],olds[0,:]),dist(new_means[1,:],olds[1,:]))
    
    olds=new_means

print("Cluster 0 Mean: "+ str(new1))
print("Cluster 1 Mean: "+ str(new2))



