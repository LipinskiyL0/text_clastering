# -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:19:55 2021

@author: Leonid
"""

import pickle


import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, ward, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

with open('rez_table_text.pkl', 'rb') as f:
    df = pickle.load(f)

Result=list(df['Q19_txt'].dropna())
# count_vectorizer = CountVectorizer()
# bag_of_words = count_vectorizer.fit_transform(Result)
# feature_names = count_vectorizer.get_feature_names()
# df_rez=pd.DataFrame(bag_of_words.toarray(), columns = feature_names)


tfidf_vectorizer = TfidfVectorizer()
values = tfidf_vectorizer.fit_transform(Result)
# Show the Model as a pandas DataFrame
feature_names = tfidf_vectorizer.get_feature_names()
df_rez=pd.DataFrame(values.toarray(), columns = feature_names)
    
feature_names = pd.DataFrame(feature_names, columns=['words'])
feature_names.to_excel('feature_names.xlsx')

stat=[]
for n_clusters in range(2, 10):
    clusterer = KMeans(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(df_rez)
    silhouette_avg = silhouette_score(df_rez, cluster_labels)
    stat.append([n_clusters,silhouette_avg ])

stat=pd.DataFrame(stat, columns=['n_clusters', 'silhouette_avg'])

plt.figure()
plt.plot(stat['n_clusters'], stat['silhouette_avg']) 
plt.title('Мешок слов')
plt.xlabel('n_clusters')
plt.ylabel('silhouette_avg')
plt.savefig('Мешок слов.png')
    
    




