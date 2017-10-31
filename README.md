# Clustering-Algorithms
Differnet Clustering Algorithms to find clusters of genes that exhibit similar expression profiles

Implementation has three clustering algorithms to find clusters of genes that exhibit similar expression profiles: 
• K-means, 
• Hierarchical Agglomerative clustering with Single Link (Min), 
• Density-based DBSCAN(Density-based spatial clustering of applications with noise)

Then, these methods are compared using external indexes namely Rand Index and Jaccard Coefficient.

*********************
****  Datasets   ****
*********************
new_dataset_1, new_dataset_2, cho.txt and iyer.txt are the two datasets


*********************
**  Dataset format **
*********************

Each row represents a gene:
1) the first column is gene_id.
2) the second column is the ground truth clusters. You can compare it with your results. "-1" means outliers.
3) the rest columns represent gene's expression values (attributes).



