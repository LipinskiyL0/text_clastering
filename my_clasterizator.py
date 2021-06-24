# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:11:52 2021

@author: Leonid
"""
import nltk
from nltk.cluster import KMeansClusterer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import matplotlib.pyplot as plt

class KMeansClusters(BaseEstimator, TransformerMixin):
    def __init__(self, k = 7):
        """
        число k определяет количество кластеров
        модель model является реализацией Kmeans
        """
        self.k = k
        self.distance = nltk.cluster.util.cosine_distance
        self.model = KMeansClusterer(self.k, self.distance, avoid_empty_clusters = True)
        
    def fit(self, documents, labels = None):
        return self
    
    def transform(self, documents):
        """
        Обучает модель K-Means векторными представлениями документов,
        полученными прямым кодированием.
        """
        return self.model.cluster(documents, assign_clusters = True)

if __name__ == '__main__':
    X1=np.random.randn(100, 1)
    X2=np.random.randn(100, 1)
    
    Y1=np.random.randn(100, 1)+5
    Y2=np.random.randn(100, 1)+5
    X=np.concatenate([X1, X2], axis=1)    
    Y=np.concatenate([Y1, Y2], axis=1)
    
    X=np.concatenate([X, Y], axis=0)
    model=KMeansClusters(k=2)
    y=model.transform(X)
    y=np.array(y)
    ind1=y==0
    ind2=y==1
    
    plt.figure()
    plt.plot(X[ind1, 0],X[ind1, 1], 'or' )
    plt.plot(X[ind2, 0],X[ind2, 1] , 'ob')
    
    
    
    