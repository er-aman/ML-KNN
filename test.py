# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:14:05 2019

@author: VISHESH
"""
import csv
import random
import math
import operator
import copy
f_size=0
k=3
a=[]
def load_resample_data(fname,training_set=[],test_set=[]):
    with open(fname,'rb') as csvfile:
        lines=csv.reader(csvfile)
        data=list(lines)
        cross_validation(data,10)
def cross_validation(data,folds=3):
    global f_size #size of fold
    dataset=list()
    data_copy=list(data)
    f_size= int(len(data)/folds)
    for i in range(folds):
        fold=list() #the formation of folds
        while len(fold)<f_size:
            index= random.randrange(len(data_copy))
            print(index)
            fold.append(data_copy.pop(index))
        dataset.append(fold)
    test_train_split(dataset,folds)
    
def test_train_split(dataset,folds):
    for f in range(folds):
        split=copy.deepcopy(dataset)
        test_split = dataset[f] # test data (test-cross validation split)
        split.remove(test_split)
        train_split=[]
        for i in range(folds-1): # handling the training data part (train- cross validation-split)
            for j in range(f_size):
                train_split.append(split[i][j])# appending the training data in train_split list
        for x in range(len(test_split)):
            for y in range(4):
                test_split[x][y]=float(test_split[x][y])
        for x in range(len(train_split)):
            for y in range(4):
                train_split[x][y]=float(train_split[x][y])
                
                
        predict(test_split,train_split)
def euclidean_dist(test,train,length):
    distance=0
    for x in range(length):
        distance += pow((test[x]-train[x]),2)
    return math.sqrt(distance)
def polynomial_kernel(test, train, length):
    p = 3
    xx = pow((1 + dot_product(test, test, length)), p)
    xy = pow((1 + dot_product(test, train, length)), p)
    yy = pow((1 + dot_product(train, train, length)), p)
    distance = math.sqrt(xx - (2 * xy) + yy)  #k(x,x)-2k(x,y)+k(y,y)
    return distance
def radial_basis_kernel(test,train,length):
    sigma = 2.3
    xx= math.exp(- pow(euclidean_dist(test,test,length),2)/pow(sigma,2))
    xy= math.exp(- pow(euclidean_dist(test,train,length),2)/pow(sigma,2))
    yy= math.exp(- pow(euclidean_dist(train,train,length),2)/pow(sigma,2))
    distance = math.sqrt(xx - (2 * xy) + yy)  #k(x,x)-2k(x,y)+k(y,y)
    return distance
def sigmoid_kernel(test,train,length):
    alpha=1
    beta=2
    xx= math.tanh(alpha*dot_product(test,test,length)+beta)
    xy= math.tanh(alpha*dot_product(test,train,length)+beta)
    yy= math.tanh(alpha*dot_product(train,train,length)+beta)
    distance = math.sqrt(xx - 2 * xy + yy)  #k(x,x)-2k(x,y)+k(y,y)
    return distance
def dot_product(d1, d2, length):
    dp = 0
    for i in range(length):
        dp += (d1[i] * d2[i])
    return dp
def get_neighbors(training_set,test_temp,k):
    distance_set = []
    length=len(test_temp)-1
    for x in range(len(training_set)):
        dist=radial_basis_kernel(test_temp,training_set[x],length)
        distance_set.append((training_set[x],dist))
    distance_set.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distance_set[x][0])
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
    match_count=0
    for x in range(len(test_set)):
        if test_set[x][-1]==predictions[x]:
            match_count+=1
    return (match_count/float(len(test_set))) * 100.0

def predict(test_set,training_set):
    #global a
    predictions=[]
    for i in range(len(test_set)):
        n=get_neighbors(training_set,test_set[i],k)
        op=get_response(n)
        predictions.append(op)
    #accuracy_percent=[]
    accuracy_percent= accuracy(test_set,predictions)
    a.append(accuracy_percent)
    print(a)
def mean_accuracy():
    mean_acc= sum(a)/len(a)
    print "Mean accuracy = " + repr(mean_acc) + "%"
def main():
   training_set=[]
   test_set=[]
   #split=0.67
   #k=3
   #predictions=[]
   load_resample_data('iris.data',training_set,test_set)
   mean_accuracy()
  # print 'Number of train samples:'+ repr(len(training_set))
   #print 'Number of test samples:' + repr(len(test_set))
         
main()    
       
       
   

        
    
