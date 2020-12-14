#Team Members: Zhaoguo Zhu/Piotr Nojeszewski
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
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


#Excel / CSV needs a heading: "Description"

class CTModel:
    """ Clustering and Topic Modelling for FOIA requests """

    def __init__(self, source, csv=False):
        """ Initialize the model.
            Input
                source [String]
                    path to the xlsx file
                (optional) csv [bool]
                    If True, the input will be read as csv
                    default: False
            Return
                None
        """
        self.source = source
        self.csv=csv

    def __setSentences(self):
        """ Read input data and set sentences
            Input
                None
            Return
                None
        """
        try:
            if(self.csv):
                self.df = pd.read_csv(self.source)
            else:
                self.df = pd.read_excel(self.source)
            self.df = self.df.fillna(method="ffill")
            self.sentences = []
            for index, row in self.df.iterrows():
                self.sentences.append(row['Description'])
        except:
            raise Exception("Error while reading the file.")

    def __setClusters(self):
        """ Set clusters
            Input
                None
            Return
                None
        """

        embedder = SentenceTransformer('distilroberta-base-paraphrase-v1')

        try:
            corpus_embeddings = embedder.encode(self.sentences)
            self.corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

            clustering_model = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward', distance_threshold=3) #, affinity='cosine', linkage='average', distance_threshold=0.4)
            clustering_model.fit(self.corpus_embeddings)
            cluster_assignment = clustering_model.labels_

            self.clustered_sentences = {}
            for sentence_id, cluster_id in enumerate(cluster_assignment):
                if cluster_id not in self.clustered_sentences:
                    self.clustered_sentences[cluster_id] = []

                self.clustered_sentences[cluster_id].append(self.sentences[sentence_id])
        except:
            raise Exception("Error while clustering.")


    def __setLabels(self):
        """ Set LDA labels
            Input
                None
            Return
                None
        """
        self.labels = {}

        try:
            for i_cluster in range(len(self.clustered_sentences)):
                count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
                doc_term_matrix = count_vect.fit_transform(self.clustered_sentences[i_cluster])
                LDA = LatentDirichletAllocation(n_components=5, random_state=42)
                LDA.fit(doc_term_matrix)
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
                        self.labels[i_cluster] = lst
                        break
                # print(f"Cluster Number {i+1}: ", exe())
                # print("Cluster Size: ", len(clustered_sentences[i]))
                # print('\n')
        except:
            raise Exception("Error in topic modelling.")


    def run(self, verbose=False):
        """ Run clustering and topic modelling.
            Input
                (optional) verbose [bool]
                    If set to True, logs will be printed.
            Return
                None
        """
        if(verbose):
            print("CTModel: Starting run()")
        self.__setSentences()
        if(verbose):
            print("CTModel: Read input data")
        self.__setClusters()
        if(verbose):
            print("CTModel: Clusters created")
        self.__setLabels()
        if(verbose):
            print("CTModel: Labales created")
            print("CTModel: Finished run()")

    def getClusters(self):
        """ Get cluster dictionary
            Input
                None
            Return
                clusters [dict]
        """
        try:
            return self.clustered_sentences
        except:
            raise Exception("First start the model with run()")

    def getLabels(self):
        """ Get LDA label dictionary
            Input
                None
            Return
                labels [dict]
        """
        try:
            return self.labels
        except:
            raise Exception("First start the model with run()")

    def saveDendrogram(self, path):
        """ Visualize Dendrogram
            Input
                @param: path [String]
                    name for the .png file
            Return
                None
        """
        try:
            assert type(path) == str
        except:
            raise Exception("path must be a string")
        try:
            self.corpus_embeddings == True
        except:
            raise Exception("First start the model with run()")
        try:
            Z = hierarchy.linkage(self.corpus_embeddings, "ward")
            dn = hierarchy.dendrogram(Z)
            plt.title("Dendrogram")
            plt.ylabel("Euclidean distances")
            plt.savefig(path)
        except:
            raise Exception("Something went wrong while trying to save file.")


# model = CTModel("2016.xlsx")
# model.run(True)
