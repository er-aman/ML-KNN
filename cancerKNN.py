#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  29 06:26:53 2019

@author: amanarora
"""
import csv
import random
import math
import operator
import copy
size_of_fold=0
k=3
accuracyList=[]
def load_resample_data(fname,training_set=[],test_set=[]):
    with open(fname,'rb') as csvfile:
        lines=csv.reader(csvfile)
        dataFromFile=list(lines)
        # Calling cross validation for k as 10 folds 
        cross_validation(dataFromFile,10)
def cross_validation(data,folds=5):
    global size_of_fold
    datframe = []
    data_copy=list(data)
    size_of_dataset = len(data)
    size_of_fold = size_of_dataset // folds
    for i in range(folds):
        fold=[]
        while len(fold)<size_of_fold:
            index= random.randrange(len(data_copy))
            # Here we get the random data lines popped up and attach to the fold
            # list till it gets to the size of the fold size which in this case 
            # is 15 as the length of the dataset is 150 and the fold size is 15
            fold.append(data_copy.pop(index))
        # Since the fold range is 10 thus with every loop we add the generated 
        # fold to data set thus the data set we have is of length 10 and size
        # of each fold is 15
        datframe.append(fold)
        # Now here we split the dataset we have obtained so far in k folds
    trainTestSplit(datframe,folds)
    
def trainTestSplit(df,folds):
    for f in range(folds):
        # Getteing the deepcoy of the data so that the changes can be made 
        # in the original file as well
        dataToSplit=copy.deepcopy(df)
        # Getting the test data
        test_data = df[f]
        # Removing the test data from the dataset
        dataToSplit.remove(test_data)
        train_data=[dataToSplit[m][n] for m in range(folds-1) for n in range(size_of_fold)]
        # Converting the string to float
        for x in range(len(test_data)):
            for y in range(0,11,1):
                # Replacing the question marks in the data set with higher values
                # Replacing outliers 
                if test_data[x][y] == '?':
                    test_data[x][y]='99999'
                test_data[x][y]=float(test_data[x][y])
        for x in range(len(train_data)):
            for y in range(0,11,1):
                if train_data[x][y] == '?':
                    train_data[x][y]='99999'
                train_data[x][y]=float(train_data[x][y])
                
        prediction(test_data,train_data)
        
def euclidean_dist(test,train,length):
    distance=0
    for x in range(1,9):
        distance += pow((test[x]-train[x]),2)
    return math.sqrt(distance)
def polynomialKernel(test, train, length):
    p = 2
    #k(x,x)-2k(x,y)+k(y,y)
    xx = pow((1 + dotProduct(test, test, length)), p)
    xy = pow((1 + dotProduct(test, train, length)), p)
    yy = pow((1 + dotProduct(train, train, length)), p)
    distance = math.sqrt(xx - (2 * xy) + yy)
    return distance
def radialBasisKernel(test,train,length):
    sigma = 2.5
    #k(x,x)-2k(x,y)+k(y,y)
    xx= math.exp(- pow(euclidean_dist(test,test,length),2)/pow(sigma,2))
    xy= math.exp(- pow(euclidean_dist(test,train,length),2)/pow(sigma,2))
    yy= math.exp(- pow(euclidean_dist(train,train,length),2)/pow(sigma,2))
    distance = math.sqrt(xx - (2 * xy) + yy)
    return distance
def sigmoidKernel(test,train,length):
    alpha=4
    beta=0.8
    #k(x,x)-2k(x,y)+k(y,y)
    xx= math.tanh(alpha*dotProduct(test,test,length)+beta)
    xy= math.tanh(alpha*dotProduct(test,train,length)+beta)
    yy= math.tanh(alpha*dotProduct(train,train,length)+beta)
    distance = math.sqrt(xx - 2 * xy + yy)
    return distance
def dotProduct(d1, d2, length):
    return sum(d1[i] * d2[i] for i in range(1,9))

def get_neighbors(training_set,test_inst,k):
    distance_set = []
    length=(1,9)
    for x in range(len(training_set)):
        dist=sigmoidKernel(test_inst,training_set[x],length)
        distance_set.append((training_set[x],dist))
    distance_set.sort(key=operator.itemgetter(1))
    neighbors=[distance_set[ynew][0] for ynew in range(k)]
    return neighbors
def get_response(neighbors):
    classvotes={}
    for x in range(len(neighbors)):
        response=neighbors[x][-1]
        if response in classvotes:
            classvotes[response]+=1
        else:
            classvotes[response]=1
    sorted_votes=sorted(classvotes.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sorted_votes[0][0]
def accuracy(test_set,predictions):
    correct=0
    for x in range(len(test_set)):
        if test_set[x][-1]==predictions[x]:
            correct+=1
    return (correct/float(len(test_set))) * 100.0

def prediction(test_data,train_data):
    predicted_result=[]
    for i in range(len(test_data)):
        nearest_neighbors=get_neighbors(train_data,test_data[i],k)
        response=get_response(nearest_neighbors)
        predicted_result.append(response)
    accuracy_percent= accuracy(test_data,predicted_result)
    accuracyList.append(accuracy_percent)
def averageAccuracy():
    avg_acc= sum(accuracyList)/len(accuracyList)
    print " Average accuracy for" + " " +repr(k) + " " + "fold cross-validation : " + repr(avg_acc) + "%"
def main():
   train_set=[]
   test_set=[]
   load_resample_data('breast-cancer-wisconsin.data',train_set,test_set)
   averageAccuracy()
         
main()    