#!/usr/bin/env python
# coding: utf-8

# In[231]:


import pandas as pd
import numpy as np
from sklearn import model_selection


# In[232]:


def step_gradient(points,learning_rate,m,c,y):
    M=len(points)
    m_slope=np.zeros((len(points[0])))
    c_slope=np.zeros((len(points[0])))
    for i in range(len(points[0])):
        for j in range(M):
            x=points[j]
            m_slope[i]+=(-2/M)*(y[j]-(m*x).sum())*x[i]
    new_m=m-learning_rate*m_slope
    return new_m
        


# In[233]:


def cost(points,m,y):
    total_cost=0
    M=len(points)
    for i in range(M):
        total_cost+=(y[i]-(m*points[i]).sum())**2
    return total_cost    


# In[234]:


def gd(points,learning_rate,num_iterations,y):
    m=np.zeros((len(points[0])))
    c=np.zeros((len(points[0])))
    for i in range(num_iterations):
        m=step_gradient(points,learning_rate,m,c,y)
        print(i,"Cost:",cost(points,m,y))
    return m           


# In[236]:


def run():
    #Loading X_train and X_test
    data=np.loadtxt(r"C:\Users\abhic\Desktop\diabates dataset\Boston data set\0000000000002417_training_boston_x_y_train.csv",delimiter=",")
    data1=np.ones((len(data),len(data[0])))
    data1[:,0:len(data[0])-1]=data[:,0:len(data[0])-1]
    # Loading x-test data
    data2=np.loadtxt(r"C:\Users\abhic\Desktop\diabates dataset\Boston data set\0000000000002417_test_boston_x_test.csv",delimiter=",")
    data3=np.ones((len(data2),len(data[0])))
    data3[:,0:len(data[0])-1]=data2[:,0:len(data[0])-1]
    y=data[:,13]
    x_train,x_test,y_train,y_test=model_selection.train_test_split(data1,y)
    learning_rate=0.1535
    num_iterations=50
    m=gd(x_train,learning_rate,num_iterations,y_train)
    print(m)
    y_pred=np.zeros((len(data3)))
    for i in range(len(data3)):
        y_pred[i]=(m*data3[i]).sum()
    np.savetxt("25Gradient.csv",y_pred) 
if __name__=="__main__":
    run()    

