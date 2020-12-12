#Author: Zhaoguo Zhu
#Partner: Piotr Nojszewski
#Clinet: Dr. Kaija Schilde
#Boston University Spark Engineering Project
#Outsourcing Security

#Citation for transformer embedding and clustering

'''
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
'''

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster import hierarchy
import nltk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


embedder = SentenceTransformer('distilroberta-base-paraphrase-v1')
df = pd.read_excel("2016.xlsx")
df = df.fillna(method="ffill")

sentence = []
for index, row in df.iterrows():
    sentence.append(row['Description'])
corpus_embeddings = embedder.encode(sentence)
corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

Z = hierarchy.linkage(corpus_embeddings, "ward")
dn = hierarchy.dendrogram(Z)
plt.title("Dendrogram")
plt.ylabel("Euclidean distances")
plt.show()
                                                        
