# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:25:54 2017

@author: Junaid
"""

import numpy as np
from sklearn.metrics import jaccard_similarity_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

text_file = 'new_dataset_2.txt'

file_values = []
with open(text_file,'r') as f:
	for line in f:
		values = line.strip().split('\t')
		file_values.append(values)

numrows = len(file_values)
numcols = len(file_values[0])
print("In", text_file, "==== row count-->",numrows,"columns count-->",numcols)
file_values=np.array(file_values).astype(np.float) # Converting to numpy array

"""
ks = file_values[:,1]
k_value = np.amax(ks)
k_value = int(k_value)  ##Value of K is largest Value of 1st Column
print("k_value :  ",k_value)
"""
k_value =3

distance_matrix=np.ndarray(shape=(numrows,numrows), dtype=float)
distance_matrix.fill(np.inf)
min_dist=np.inf
cluster1=0
cluster2=0
clusters=[]

##initially each cluster has a single row as its cluster 
for i in range(numrows):
    clusters.append([file_values[i,0]-1])
    

for i in range(numrows):
    for j in range(i+1,numrows):
        distance_matrix[j][i]=np.linalg.norm((file_values[i,2:])-(file_values[j,2:]))



num_iter=0
while(len(distance_matrix)> k_value):
    num_iter+=1
    min_dist=np.amin(distance_matrix)
    cluster2,cluster1=(np.unravel_index(np.argmin(distance_matrix),distance_matrix.shape))
    
    if(cluster2<cluster1):
        print("Something Went Wrong : Check")
        temp=cluster1
        cluster1=cluster2
        cluster2=temp


    for i in range(len(distance_matrix)):
        if(i!=cluster1 and i!=cluster2):
            if(i<cluster1):
                dist1=distance_matrix[cluster1][i]
            else:
                dist1=distance_matrix[i][cluster1]
            if(i<cluster2):
                dist2=distance_matrix[cluster2][i]
            else:
                dist2=distance_matrix[i][cluster2]
            
            if(i<cluster1):
                distance_matrix[cluster1][i]=min(dist1,dist2)
            else:
                distance_matrix[i][cluster1]=min(dist1,dist2)
            
    distance_matrix=np.delete(distance_matrix,cluster2,axis=0)  ##delete ro
    distance_matrix=np.delete(distance_matrix,cluster2,axis=1)  ##delete column


    clusters[cluster1]=clusters[cluster1]+clusters[cluster2]   
    
    ##removre this entry from Cluster as its mergerd with cluster1
    del clusters[cluster2]


pred_labels = np.zeros(numrows)
for i in range(len(clusters)):
    for j in range(len(clusters[i])):
        pred_labels[int(clusters[i][j])]= i


groundTruth = np.zeros(shape=(numrows,numrows))
predicted = np.zeros(shape=(numrows,numrows))

for i in range(len(pred_labels)):
    for j in range(len(pred_labels)):
        if (pred_labels[i] == pred_labels[j]):
            predicted[i][j]=1
            predicted[j][i]=1
                     
for i in range(len(groundTruth)):
    for j in range(len(groundTruth)):
        if (file_values[i][1] == file_values[j][1]):
            groundTruth[i][j]=1
            groundTruth[j][i]=1
                       


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



ValidData = np.ndarray(shape=(numrows,numcols-2),dtype=float)
for i in range(numrows):
    ValidData[i,:]=file_values[i,2:]
    
    
pca = PCA(n_components=2)
pca.fit(ValidData)
scores = pca.transform(ValidData)


f, ax = plt.subplots(figsize=(10,5))
labelset=set(pred_labels)

for name in labelset:
    x = scores[pred_labels[:]==name,0]  #all places where label is met and then plot (1st eigen*x) as x axis 
    y = scores[pred_labels[:]==name,1]  #all places where label is met and then plot (2nd eigen*x) as y axis
    cluster_id = "Cluster_Id: "+ str(int(name))
    ax.scatter(x, y,marker='o',label=cluster_id) ## marker='o'

plt.title("Hierarchical PCA plot for : "+str(text_file))
plt.legend(loc='upper right',ncol=1,fontsize=12)
plt.show()
