#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:56:07 2022

@author: liuyilouise.xu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_fin  = pd.read_csv("FinlandWhole.txt", header=None, names=['X','Y'])
data_joen  = pd.read_csv("JoensuuRegion.txt", header=None, names=['X','Y'])

#X_fin = data_fin['X']
#Y_fin = data_fin['Y']

#X_joen = data_joen['X']
#Y_joen = data_joen['Y']

def assign_labels(X, Y, C):
    
    labels = np.zeros(X.size)
    
    
    for i in range(X.size):
        
        # compute min distance to each centroid
        min_distance = float('inf')
        for c in C:
            distance = np.hypot(X[i]-X[c], Y[i]-Y[c])
            if distance < min_distance:
                min_distance = distance
                min_cluster = c
                
        labels[i] = min_cluster
    
    return labels

def compute_centroids(X, Y, C, labels):
    
    new_centroids = []
    for c in C:
        
        # compute mean of cluster
        X_mean = np.mean(X[labels == c])
        Y_mean = np.mean(Y[labels == c])
        
        # find closest data pt to mean of cluster
        min_distance = float('inf')
        new_centroid = c
        for i, (x, y) in enumerate(zip(X[labels == c], Y[labels == c])):
            distance = np.hypot(x-X_mean, y-Y_mean)
            if distance < min_distance:
                min_distance = distance
                new_centroid = X.index[labels==c][i]
        
        new_centroids.append(new_centroid) 
        
    return np.asarray(new_centroids)

def compute_initial_centroids(X,Y, NUM_CENTROIDS):
    C = np.zeros(NUM_CENTROIDS)
    min_distance_matrix = np.ones(shape = (X.size,NUM_CENTROIDS)) # row = data, col = centroid
    for i, c in enumerate(C):
        if i==0:
            C[i] = np.random.choice(np.shape(X)[0])
            for j in range(X.size):
                min_distance_matrix[j, i] = np.hypot(X[j]-X[C[i]], Y[j]-Y[C[i]])**2       

        else:
            # compute next centroid from previous min distance
            next_centroid = np.where(min_distance_matrix[:,i-1] == max(min_distance_matrix[:,i-1]))
            C[i] = next_centroid[0]
            
            # update min distance matrix
            # don't need to update on last iteration
            if i!=(NUM_CENTROIDS-1):
                for j in range(X.size):
    
                    #distance of data j with previous centroid i
                    distance = np.hypot(X[j]-X[C[i]], Y[j]-Y[C[i]])**2
                    
                    #min distance of data j up till i centroids
                    min_distance_matrix[j, i] = min(distance, min_distance_matrix[j, i-1]) 
    return C

# computer mean cluster distance for unlabeled data
def compute_distance(X, Y, labels, C, ):
    distance = 0
    for c in C:
        X_cluster = X[labels==c]
        Y_cluster = Y[labels==c]
        for x, y in zip(X_cluster, Y_cluster):
            distance += np.hypot(x - X[c], y - Y[c])
    return distance / X.size           

# outputs list of mean cluster distance
# n - number of iterations
# k - paramter for k means
def kmeans(X, Y, k, n):
    distances = []
    C = compute_initial_centroids(X, Y, k)
    
    for it in range(n):
        
        # assign data points to cluster
        labels  = assign_labels(X, Y, C)  
        distances.append(compute_distance(X, Y, labels, C))
         
        # recompute centroids
        C = compute_centroids(X, Y, C, labels)           
        #print("new centroid:", C)
        
    distances.append(compute_distance(X, Y, labels, C))
    return C, distances

# only outputs final mean cluster distance
def kmeans_final(X, Y, k, n):
    C = compute_initial_centroids(X, Y, k)
    
    for it in range(n):
        
        # assign data points to cluster
        labels  = assign_labels(X, Y, C)  
         
        # recompute centroids
        C = compute_centroids(X, Y, C, labels)           
        #print("new centroid:", C)
        
    distance = compute_distance(X, Y, labels, C)
    return C, distance



def lab_plots(data, ITERATIONS, name):
    X = data['X']
    Y = data['Y']
    # part 1
    
    C, distances = kmeans(X, Y, k=4, n=ITERATIONS)
    print(C)
    
    plt.xlabel('x')
    plt.xlabel('y')
    plt.title(f'Clustering for {name}')
    
    #plot data (black) and centroids (yellow)
    X_norm = np.true_divide(X,max(X))
    Y_norm = np.true_divide(Y,max(Y))
    
    plt.scatter(X_norm,Y_norm)
    plt.scatter(X_norm[C],Y_norm[C],c='r')
    plt.show()
    
    plt.xlabel('iterations')
    plt.ylabel('mean cluster distance')
    plt.title(f'Clustering MCD for k=4, {name}')
    plt.scatter(range(len(distances)), distances)
    plt.show()
    
    # part 2
    
    best_dist = min(distances)
    ks = [4, 8, 12, 20]
    k_distances = []
    for k in ks:
        C, distance = kmeans_final(X, Y, k=k, n=ITERATIONS)
        print("k: ", k, "MCD: ", distance)
        k_distances.append(distance)
        if distance < best_dist:
            #print("Better k: ", k, "centroids: ", C)
            best_dist = distance
    print(C)
    
    plt.xlabel('k')
    plt.ylabel('mean cluster distance')
    plt.title(f'Clustering MCD for various k, {name}')
    plt.scatter(ks, k_distances)
    plt.show()
        
ITERATIONS = 10
lab_plots(data_fin, ITERATIONS, "Finland")

lab_plots(data_joen, ITERATIONS, "Joensuu Region")