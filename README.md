# Sentence Clustering: Clustering DOD documents using Sentence Transformers / Hierarchical Agglomerative Algorithm / LDA

This repository provides an easying method of clustering sentences based on semantic similarity. Sentences are embedded using transformers, clustered by hierarchical agglomerative algorithm and eventually each cluster is given name based on repeat of important words by LDA. To apply our program, you will be only using file in the 'src' folder. Other files in the repository are for testing and research purposes.


Original project idea came from Dr. Kaija Schilde and the process was supervized under Professor Dharmesh Tarapore from the Boston University Spark Engineering Community. @Professor Tarapore: for grading purposes, please also only look at the files at the 'src' folder and follow the guideline below.


The contents of the code orginates from the UKPLab, please refer to our citation and links in the files and below.




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

The input of the file has to be either csv or xlsx with a column name "Description." You will need to go to both Dendrogram_Visualization.py and Clustering_Topic_Model.py and modify the pandas file read line to the format you want.

**For CSV**
```
df = pd.read_csv(<filename>)
```

**For XLSX**
```
df = pd.read_excel(<filename>)
```

## Dendrogram Visulization

Before this section, we highly recommend that you watch the following video from youtube: https://www.youtube.com/watch?v=ijUMKMC4f9I. It tells you how to read the dendrogram, which in our case the distance measurement is Euclidean Distance. You can choose your own distance threshold based on how many cluster you want to have. This distance threshold number will be used for the next step.


## 
