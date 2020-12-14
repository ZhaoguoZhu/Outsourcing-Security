# Sentence Clustering: Clustering DOD documents using Sentence Transformers / Hierarchical Agglomerative Algorithm / LDA

This repository provides an easying method of clustering sentences based on semantic similarity. Sentences are embedded using transformers, clustered by hierarchical agglomerative algorithm and eventually each cluster is given name based on repeat of important words by LDA. To apply our program, you will be only using file in the 'src' folder. Other files in the repository are for testing and research purposes.


Original project idea came from Dr. Kaija Schilde and the process was supervized under Professor Dharmesh Tarapore from the Boston University Spark Engineering Community. @Professor Tarapore: for grading purposes, please also only look at the files at the 'src' folder and follow the guideline below.


The contents of the code orginates from the UKPLab, please refer to our citation and links in the files and below.

**Original Code from UKPLab**

https://github.com/UKPLab/sentence-transformers




## Installation
To properly run the code we provide, please Install the following libraries with pip.

```
pip install sentence-transformers scikit-learn scipy numpy matplotlib ipython pandas nltk
```

```
pip install xlrd==1.2.0
```


## Quick Start
Run python commands below for a quick demo of the clustering and topic modeling on a 2006 FOIA logs from DOD:

```
from CTModel import CTModel
x=CTModel('2006.xlsx')
x.run(True)
x.getLabels()
```

Run python commands below to see demo of contract data reader:

```
from Contracts import Contracts
y=Contracts('web2005.txt')
y.run(True)
y.getSortedLabels[:10]
```

## Docker
To run the code in docker container go to /src folder. Then build the into ctmodel image and run interactive mode. Files will be in /tmp/.

**On UNIX**
```console
$ sudo docker build -t ctmodel .
$ sudo run -it ctmodel bash
```
**In Docker Container**
```console
root@docker:/# cd /tmp/
root@docker:/tmp# python
```

## CTModel Initialization
We will start with clustering and topic modeling for FOIA requests (CTModel). Input file has to be either csv or xlsx with request descriptions in a column named "Description." The "Description" column should contain English sentences for each row.

```
from CTModel import CTModel

```
**For CSV**
```
x=CTModel('Foia.csv', True)
```

**For XLSX**
```
x=CTModel('Foia.xlsx')
```

## CTModel Run
After initializing the model with an input file, we can start it with run().
```
x.run()
```

**To run with logs**
```
x.run(True)
```

## CTModel Output
After a successful run, we get the output of the clustering and topic modeling using two methods:

**Get Clusters**
```
x.getClusters()
```
**Get LDA Labels**
```
x.getLabels()
```

## CTModel Dendrogram Visulization

Before this section, we highly recommend that you watch the following video from youtube: https://www.youtube.com/watch?v=ijUMKMC4f9I. It tells you how to read the dendrogram, which in our case the distance measurement is Euclidean Distance. After a successful CTModel run(), execute function saveDendrogram(<filename>) - and you will see how the cluster can be if you choose difference distance threshold (the number on the left). You can choose your own distance threshold based on how many cluster you want to have. This distance threshold number will be used for the next step.


## CTModel Clustering and Topic Modeling

Edit Clustering_Topic_Model.py and get to line:

```
clustering_model = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', distance_threshold=3)
```

Change the distance threshold to the number you like. Then you are all set. What you will get are the clusters and the size of those clusters which represent how many number of sentences are in those cluster. Each Cluster will have 3 Topic name that represent what this cluster is mostly likely talking about.

To see the sentences within the cluster, get clusters with getClusters() and see contents of an individual cluster (e.g. 0):

```
clusters = x.getClusters()
clusters[0]
```

## Contracts Initialization
To initialize the Contracts run the commands below. The program is meant to work with the DCADS fiscal year contract data, which can be imported from 'Defense Contract Action Data System Links.docx' in root folder.

```
from Contracts import Contracts
y = Contracts('2006.txt')
```

## Contracts Run
After initializing the model with an input file, we can start it with run().
```
y.run()
```

**To run with logs**
```
y.run(True)
```

## Contracts Output
After a successful run, we get the output label dictionary with occurrence count, and a sorted list of occurrences as an array:

**Get Labels**
```
x.getLabels()
```
**Get Sorted Labels**
```
x.getSortedLabels()
```
