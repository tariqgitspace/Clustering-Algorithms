# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 23:17:52 2017

@author: Junaid
"""

from multiprocessing import Queue
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


q = Queue()
                    
def DBSCAN(file_values, eps, MinPts):
    counter_Idx = 0
    numrows = len(file_values)
    cluster=np.zeros(numrows)
    cluster.fill(-1)
    visited=np.zeros(numrows)
    core_border_noise = ["" for x in range(numrows)]


    counter_Idx = 0
    for root in range(numrows):
        if(visited[root]==0):
            visited[root] =1
            NeighborPts = regionQuery(root, eps,numrows,file_values)  ##all neighbours within eps distance
            if (len(NeighborPts) < MinPts):
                core_border_noise[root] = "Noise"
            else:
                counter_Idx +=1
                core_border_noise[root] = "Core"
                expandCluster(root, NeighborPts, counter_Idx, eps, MinPts,cluster,visited,core_border_noise)

    Jaccard(file_values,cluster)
    plot(file_values,numrows,numcols,cluster)
    

def plot(file_values,numrows,numcols,cluster):
    ValidData = np.ndarray(shape=(numrows,numcols-2),dtype=float)
    for i in range(numrows):
        ValidData[i,:]=file_values[i,2:]
        
        
    pca = PCA(n_components=2,random_state=7)
    pca.fit(ValidData)
    scores = pca.transform(ValidData)
    
    
    f, ax = plt.subplots(figsize=(10,5))
    labelset=set(cluster)
    
    for name in labelset:
        x = scores[cluster[:]==name,0]  #all places where label is met and then plot (1st eigen*x) as x axis 
        y = scores[cluster[:]==name,1]  #all places where label is met and then plot (2nd eigen*x) as y axis
        cluster_id = "Cluster_Id: "+ str(int(name))
        ax.scatter(x, y,marker='o',label=cluster_id) ## marker='o'
    
    plt.title("DBSCAN PCA plot for : "+str(text_file))
    plt.legend(loc='upper left',ncol=1,fontsize=12)
    plt.show()

def expandCluster(root, NeighborPts, counter_Idx, eps, MinPts,cluster,visited,core_border_noise):
    global q
    
    cluster[root] = counter_Idx

    for neigh in NeighborPts:
        q.put(neigh)
    
    while(not q.empty()):
        neigh = q.get()
        if visited[neigh]==0:
            visited[neigh]=1
            MoreNeighborPts = regionQuery(neigh, eps,numrows,file_values)
            if (len(MoreNeighborPts) >= MinPts) :
                core_border_noise[neigh] = "Core"
                for x in MoreNeighborPts:
                    q.put(x)
            else:
                core_border_noise[neigh] = "Noise"
        if (cluster[neigh]== -1):
            cluster[neigh] = counter_Idx
         

def regionQuery(root, eps,numrows,file_values):
    NeighborPts =[]
    for i in range(numrows):
        distance = np.linalg.norm((file_values[i,2:])-(file_values[root,2:]))
        if (distance <= eps):
            NeighborPts.append(i)    
    
    return NeighborPts

def Jaccard(file_values,cluster):
    groundTruth = np.zeros(shape=(numrows,numrows))
    predicted = np.zeros(shape=(numrows,numrows))
    
    for i in range(len(groundTruth)):
        for j in range(len(groundTruth)):
            if (file_values[i][1] == file_values[j][1]):
                groundTruth[i][j]=1
                groundTruth[j][i]=1
                           
    for i in range(len(cluster)):
        for j in range(len(cluster)):
            if (cluster[i] == cluster[j]):
                predicted[i][j]=1
                predicted[j][i]=1
    
    same1Count=0
    same0Count=0
    diff=0
    for i in range(len(predicted)):
        for j in range(len(predicted)):
            if groundTruth[i][j] ==1:
                if groundTruth[i][j] == predicted[i][j]:
                    same1Count +=1
                else :
                    diff +=1
            else:
                if groundTruth[i][j] == predicted[i][j]:
                    same0Count +=1
                else:
                    diff +=1
                
    print(diff)
    Jaccard = (same1Count)/(same1Count+diff)        
    print("Jaccard : ",Jaccard)
    
    rand = (same0Count + same1Count)/(numrows*numrows)
    print("rand : ",rand)

    unique = []
    for x in cluster:
        if (not(x in unique)):
            unique.append(x)
    
    print("Number of cluster :  ",len(unique))
    print(unique)
    
if __name__=="__main__":
    text_file = 'new_dataset_1.txt'
    
    file_values = []
    with open(text_file,'r') as f:
    	for line in f:
    		values = line.strip().split('\t')
    		file_values.append(values)
    
    numrows = len(file_values)
    numcols = len(file_values[0])
    print("In", text_file, "==== row count-->",numrows,"columns count-->",numcols)
    file_values=np.array(file_values).astype(np.float) # Converting to numpy array
    eps = 1.03
    MinPts = 4
    DBSCAN(file_values,eps, MinPts)