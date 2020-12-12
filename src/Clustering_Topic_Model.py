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
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

embedder = SentenceTransformer('distilroberta-base-paraphrase-v1')
df = pd.read_excel("2016.xlsx")
df = df.fillna(method="ffill")

sentence = []
for index, row in df.iterrows():
    sentence.append(row['Description'])
corpus_embeddings = embedder.encode(sentence)
corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

clustering_model = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', distance_threshold=3) #, affinity='cosine', linkage='average', distance_threshold=0.4)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []

    clustered_sentences[cluster_id].append(sentence[sentence_id])

length = len(clustered_sentences)

def exe():
    D = {}
    for i,topic in enumerate(LDA.components_):
        x = [count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]]
        for j in x:
            if j in D:
                D[j] += 1
            else: D[j] = 1
    Sorted = sorted(D.items(), key=lambda x: x[1], reverse=True)
    lst = []
    Useless_Word = ["requesting","request", "requests", "copy", "copies", "documents", "document", "information", "informations", "department", "departments", "records", "record", "reports", "report",'pertaining', 'concerning', 'regarding','list'] 
    for i in Sorted:
        if i[0] in Useless_Word:
            None
        else:
            lst.append(i[0])
        if len(lst) == 3:
            return lst

for i in range(length):
    count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
    doc_term_matrix = count_vect.fit_transform(clustered_sentences[i])
    LDA = LatentDirichletAllocation(n_components=5, random_state=42)
    LDA.fit(doc_term_matrix)
    print(f"Cluster Number {i+1}: ", exe())
    print("Cluster Size: ", len(clustered_sentences[i]))
    print('\n')
