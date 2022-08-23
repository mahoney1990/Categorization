# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:54:00 2022

@author: mahon

"""
#The goal of this script is to create a simple SVM from scratch
#We will be looking at a linear SVM with only two dimensions to keep the 
#Problem simple


import numpy as np
import sklearn
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


#Define Euclidean distance operator
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

#Test it out
x=[1,1]
y=[6,5]

dist(x,y)



#Define distance from point to line operator (in 2 space), m=slope, b=intercept
def line_dist(point,m,c):
    n=len(point)
    
    if n>2:
        print('Array must be 2-dimensional')
    
    #Extract line coefficiewnts
    a=m
    b=-1
    c=c
    
    #Apply distance formula
    dist=(a*point[0]+b*point[1]+c)/(a**2+b**2)**(1/2)    
    return dist

#Test it out
line_dist(x,1,0)
line_dist(y,1,0)

#Goal of SVM: pick parameters m (slope) and c (intercept) so that points above the 
#line y=mx+c are categorized as type 0 points below are type 1. We examine points close to the 
#Line on each iteration of the optimization routine

#Lets start by building an example datset where the bifurcation between type 0 and
#Type 1 is like super obvious


  
# creating datasets X containing n_samples
# Y containing two classes
band=1

X, Y = make_blobs(n_samples=20, centers=2,
                  random_state=0, cluster_std=0.40)

# plotting scatters 
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring');
plt.show() 

#Calculate the closest five points to the line y=x
point_dist=[]
candidate_dist=[]

n=len(Y)
num_candidates=5

#Define slope and intercept terms
m,c=1,0

for i in range(n):

    pt=np.array([X[i,0],X[i,1]])
    category=Y[i]
    
    point_dist.append(line_dist(pt,m,c))




