# Sentence Clustering: Clustering DOD documents using Sentence Transformers / Hierarchical Agglomerative Algorithm / LDA

This repository provides an easying method of clustering sentences based on semantic similarity. Sentences are embedded using transformers, clustered by hierarchical agglomerative algorithm and eventually each cluster is given name based on repeat of important words by LDA. To apply our program, you will be only using file in the 'src' folder. Other files in the repository are for testing and research purposes. 


Original project idea came from Dr. Kaija Schilde and the process was supervized under Professor Dharmesh Tarapore from the Boston University Spark Engineering Community. @Professor Tarapore: for grading purposes, please also only look at the files at the 'src' folder and follow the guideline below.


The contents of the code orginates from the UKPLab, please refer to our citation and links in the files and below.

**Original Code from UKPLab**

https://github.com/UKPLab/sentence-transformers




## Installation
To properly run the code we provide, please Install the following libraries with pip.

```
pip install -U sentence-transformers
```

```
pip install -U scikit-learn
```

```
python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
```

```
pip install ntlk
```
```
python install pandas
```


## Getting Started

The input of the file has to be either csv or xlsx with a column name "Description." Under the "Description" should english sentences for each roll. You will need to go to both Dendrogram_Visualization.py and Clustering_Topic_Model.py and modify the pandas file read line to the format you want.

**For CSV**
```
df = pd.read_csv(<filename>)
```

**For XLSX**
```
df = pd.read_excel(<filename>)
```

## Dendrogram Visulization

Before this section, we highly recommend that you watch the following video from youtube: https://www.youtube.com/watch?v=ijUMKMC4f9I. It tells you how to read the dendrogram, which in our case the distance measurement is Euclidean Distance. Compile the Dendrogram_Visualization.py and you will see how the cluster can be if you choose difference distance threshold (the number on the left). You can choose your own distance threshold based on how many cluster you want to have. This distance threshold number will be used for the next step.


## Clustering and Topic Modeling

Edit Clustering_Topic_Model.py and get to line:

```
clustering_model = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', distance_threshold=3)
```

Change the distance threshold to the number you like. Then you are all set. Compile file Clustering_Topic_Model.py. What you will get are the clusters and the size of those clusters which represent how many number of sentences are in those cluster. Each Cluster will have 3 Topic name that represent what this cluster is mostly likely talking about.

To see the sentences within the cluster, please type in the shell:

```
clustered_sentences[<# of the cluster minus 1>]
```

For instance, if you want to see content of cluster 1, what you actually need to type is clustered_sentences[0]



