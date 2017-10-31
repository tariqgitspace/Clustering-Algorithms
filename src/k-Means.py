import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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

ks = file_values[:,1]
k_value = np.amax(ks)
k_value = int(k_value)


defaultCentroids = False

if defaultCentroids:
    IterationThreshold =50
    centroids = np.ndarray(shape=(k_value,numcols-2),dtype=float)
    for i in range(k_value):
        centroids[i,:]=file_values[i,2:]
else:
    k_value =3
    IterationThreshold =10
    centroids = np.ndarray(shape=(k_value,numcols-2),dtype=float)
    
    centroid_input = [3,5,9]
    for i in range(len(centroid_input)):
        row = centroid_input[i]
        centroids[i,:]=file_values[row,2:]



#print("mean_arr",mean_arr)
prev_centroids = np.ndarray(shape=(k_value,numcols-2),dtype=float)

num_iter = 0

while(True):
    prev_centroids[:,:]=centroids[:,:]
    #print("prev_mean",prev_mean)
    cluster_id_distance = np.ndarray(shape=(numrows,2))
    cluster_id_distance.fill(np.inf)
    for i in range(len(file_values)):  ##rows of file
        for j in range(k_value):  # distance with respect to each cluster
            distance = np.linalg.norm((centroids[j,:])-(file_values[i,2:])) ##distance between different clusters and rows   
            if(distance<cluster_id_distance[i][1]): ##distance less than previous distance
                cluster_id_distance[i][0]=j
                cluster_id_distance[i][1]=distance #update
                
    ##update cluster row as mean of all inputs in that cluster
    ##means multiple times ?
    for i in range(k_value):
        x = np.where(cluster_id_distance[:,0] == i)
        x = np.concatenate(x)  #flatten array
        centroids[i] = np.nanmean(file_values[x,2:],axis=0)


    if(np.array_equal(centroids,prev_centroids)):
        break
    num_iter+=1  #Tariq: Moved at end
    if(num_iter > IterationThreshold):
        break;
    


groundTruth = np.zeros(shape=(numrows,numrows))
predicted = np.zeros(shape=(numrows,numrows))

for i in range(len(groundTruth)):
    for j in range(len(groundTruth)):
        if (file_values[i][1] == file_values[j][1]):
            groundTruth[i][j]=1
            groundTruth[j][i]=1
                       
for i in range(len(predicted)):
    for j in range(len(predicted)):
        if (cluster_id_distance[i][0] == cluster_id_distance[j][0]):
            predicted[i][j]=1
            predicted[j][i]=1

same1Count=0
same0Count=0
diff=0
for i in range(len(predicted)):
    for j in range(len(predicted)):
        if groundTruth[i][j] ==1:
            if groundTruth[i][j] ==predicted[i][j]:
                same1Count +=1
            else :
                diff +=1
        else:
            if groundTruth[i][j] ==predicted[i][j]:
                same0Count +=1
            else:
                diff +=1
            
#different  = (numrows*numrows) - same0Count-same1Count
 
#print(different)
#print(diff)
Jaccard = (same1Count)/(same1Count+diff)        
print("Jaccard : ",Jaccard)

rand = (same0Count + same1Count)/(numrows*numrows)
print("rand : ",rand)

print("No. of iterations: ",num_iter)



ValidData = np.ndarray(shape=(numrows,numcols-2),dtype=float)
for i in range(numrows):
    ValidData[i,:]=file_values[i,2:]
    
    
pca = PCA(n_components=2)
pca.fit(ValidData)
scores = pca.transform(ValidData)


f, ax = plt.subplots(figsize=(10,5))
labels=cluster_id_distance[:,0]
labelset=set(labels)

for name in labelset:
    x = scores[cluster_id_distance[:,0]==name,0]  #all places where label is met and then plot (1st eigen*x) as x axis 
    y = scores[cluster_id_distance[:,0]==name,1]  #all places where label is met and then plot (2nd eigen*x) as y axis
    cluster_id = "Cluster_Id: "+ str(int(name))
    ax.scatter(x, y,marker='o',label=cluster_id) ## marker='o'

plt.title("K-Means PCA plot for : "+str(text_file))
plt.legend(loc='upper right',ncol=1,fontsize=12)
plt.show()
