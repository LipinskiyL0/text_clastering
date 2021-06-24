# -*- coding: utf-8 -*-
"""
Created on Fri May 28 00:27:56 2021

Здесь реализуется класс по кластеризации корпуса текста с помощью генетического алгоритма
text - корпус текста
v_metrics - выбранный способ векторизации текста. По умолчанию bw - мешок слов
            альтернатива - tfidef

self.best_clusterer - запоминаем наилучший кластеризатор
self.best_er - запоминаем наилучшее значение критерия
self.best_ind - запоминаем наилучшие столбцы, которые соответствуют self.clusterer и self.er

self.ind - столбцы, которые вернул ГА
self.er - значение критерия, которые вернул ГА. Т.к. k-means строится случайно, то это может не соответствовать
            self.best_er и self.best_ind

my_ga_clastering - проводит кластеризацию текста. На входе адаптированный текст
                    на выходе кластеры

clastering_text - управляет кластеризацией текста. Является надстройкой для
                  my_ga_clastering. На вход df состоит из двух столбцов. 
                  Первый столбец исходный текст, второй - адаптированный текст.
                  Столбец с адаптированным текстом содержит в названии 'txt'
                
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
from sklearn.cluster import DBSCAN

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
from GAClass import GAClassMy
from sklearn.decomposition import LatentDirichletAllocation

class my_ga_clastering:
    def __init__(self, text, v_metrics='bw'):
        self.text=text
        self.v_metrics=v_metrics
        if v_metrics=='tfidf':
            tfidf_vectorizer = TfidfVectorizer()
            values = tfidf_vectorizer.fit_transform(text)
            # Show the Model as a pandas DataFrame
            feature_names = tfidf_vectorizer.get_feature_names()
            df_rez=pd.DataFrame(values.toarray(), columns = feature_names)
        elif v_metrics=='bw':
            count_vectorizer = CountVectorizer()
            bag_of_words = count_vectorizer.fit_transform(text)
            feature_names = count_vectorizer.get_feature_names()
            df_rez=pd.DataFrame(bag_of_words.toarray(), columns = feature_names)
            ind=df_rez.sum(axis=0)
            ind=ind>2
            df_rez=df_rez.loc[:, ind]
        elif v_metrics=='LDA':
            count_vectorizer = CountVectorizer()
            bag_of_words = count_vectorizer.fit_transform(text)
            feature_names = count_vectorizer.get_feature_names()
            
            
            
            lda = LatentDirichletAllocation(n_components=10, max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

            df_rez=lda.fit_transform(bag_of_words)
            df_rez=pd.DataFrame(df_rez)
            
            
            
        else:
            print("Ошибочная метрика")
            return
        
        self.df_rez=df_rez
        self.best_er=-1


        return
    
    def er_clastering(self, ind, flag=0):
        #исследуем эффективность словаря 
        #ind - бинарная строка от GA кодирующая включение/исключение признака
        #flag - если =0 то вычисление пригодности иначе мы вычсляем метки и количество кластеров
        ind1=np.array(ind)
        ind1=ind1.astype(bool)
        df_rez=self.df_rez.loc[:, ind1]
        
        stat=[]
        for n_clusters in range(5, 10):
            clusterer = KMeans(n_clusters=n_clusters, max_iter=1000)
            cluster_labels = clusterer.fit_predict(df_rez)
            # clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            # cluster_labels = clusterer.fit_predict(df_rez)
            # clusterer = DBSCAN()
            # cluster_labels = clusterer.fit_predict(df_rez)
            silhouette_avg = silhouette_score(df_rez, cluster_labels)
            stat.append([n_clusters,silhouette_avg ])
            if self.best_er<silhouette_avg:
                self.best_clusterer=clusterer
                self.best_er=silhouette_avg
                self.best_ind=ind.copy()
                self.best_cluster_labels=cluster_labels.copy()
                self.best_n_clusters=n_clusters
                
        stat=pd.DataFrame(stat, columns=['n_clusters', 'silhouette_avg'])
        if flag!=0:
            i=stat['silhouette_avg'].argmax()
            n_clusters=stat['n_clusters'].iloc[i]
            
            df_rez=self.df_rez.loc[:, ind1]
            clusterer = KMeans(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(df_rez)
            silhouette_avg = silhouette_score(df_rez, cluster_labels)
            print(stat)
            return (n_clusters, cluster_labels, silhouette_avg)
        
        return stat['silhouette_avg'].max()
    
    def er_clastering_best(self):
        #рассчитываем метки на лучшем кластеризаторе и оцениваем критерий силуэта
        ind1=np.array(self.best_ind)
        ind1=ind1.astype(bool)
        df_rez=self.df_rez.loc[:, ind1]
        
        cluster_labels = self.best_clusterer.predict(df_rez)
        silhouette_avg = silhouette_score(df_rez, cluster_labels)
            
        
        return silhouette_avg
    
    def opt(self):
        ga=GAClassMy()
        flag=ga.inicialization(self.er_clastering, n_bits=self.df_rez.shape[1], p_mut=1/self.df_rez.shape[1],
                               n_tur=5, n_ind=20)
        if flag==False:
            print("Ошибка инициализации GA")
            return False
        
        ind, er=ga.opt( n_iter=5)
        self.ind=ind
        self.er=er
        return ind, er

    
class clastering_text:  

    def __init__(self, df, v_metrics='tfidf'):
        #df[name] - столбец с исходным текстом
        #df[name_txt] - столбец с адаптивным текстом
        
        #создаем индекс 
        cols=list(df.columns)
        if len(cols)!=2:
            print('Ошибка входных данных')
            return
        
        if ('txt' in cols[0])==False:
            name=cols[0]
            if name+'_txt'!=cols[1]:
                print('Ошибка входных данных')
                return 
        elif ('txt' in cols[1])==False:
            name=cols[1]
            if name+'_txt'!=cols[0]:
                print('Ошибка входных данных')
                return 
        else:
            print('Ошибка входных данных')
            return
        
        self.df=df.copy()
        
        ind=df[name+'_txt'].notnull()
        self.df = self.df.loc[ind, :]
        self.df['index']=np.arange(len(self.df))
        self.df=self.df[['index']+cols]
        self.v_metrics=v_metrics
        self.name=name
        return
        
    def clastering(self, level=0, claster={0:0}):
        #проводим кластеризацию по заданному уровню
        #Если уровень = 0 то проводим кластеризацию по всей выборке
        #иначе берем подвыборку согласно claster. 
        #claster представляет собой словарь level:номер кластера
        #например claster={0:1, 1:2} выбираем подвыборку так, что нулевой уровень
        #кластеризации и первый кластер, из них на первом уровне кластеризации второй кластер
        
        
        if level==0:
            clastering=my_ga_clastering(list(self.df[name+'_txt']),v_metrics = self.v_metrics)
            ind, er=clastering.opt()
            self.df['level'+'_0']=clastering.best_cluster_labels
        else:
            flag=0
            for k in claster:
                if flag==0:
                    flag=1
                    ind=self.df['level'+'_'+str(k)]==claster[k]
                else:
                    ind1=self.df['level'+'_'+str(k)]==claster[k]
                    ind=ind1&ind
            df1=self.df[ind].copy()
            clastering=my_ga_clastering(list(df1[name+'_txt']),v_metrics = self.v_metrics)
            ind, er=clastering.opt()
            df1['level'+'_'+str(level)]=clastering.best_cluster_labels
            df1=df1[['index', 'level'+'_'+str(level)]]
            self.df=pd.merge(self.df, df1, how='outer',left_on='index', right_on='index')
            
        return
    
if __name__ == '__main__':
    with open('rez_table_text.pkl', 'rb') as f:
        df  = pickle.load(f)
    name='Q18'
    # ind=df[name+'_txt'].notnull()
    # r=df[name].loc[ind]
    # Result=list(df[name+'_txt'].loc[ind])
    # clastering=my_ga_clastering(Result,v_metrics='tfidf' )
    # ind, er=clastering.opt()
    # rez=clastering.er_clastering(ind, flag=1)
    # rez_best=clastering.er_clastering_best()
    # with open(name+'_txt.pkl', 'wb') as f:
    #     pickle.dump(clastering, f)
    
    
    # rez=pd.DataFrame({'text0':r,'text':Result, 'label':clastering.best_cluster_labels})
    # rez.to_excel(name+'_txt.xlsx')
    # print(rez['label'].value_counts())
    cl=clastering_text(df[[name, name+'_txt']], v_metrics='tfidf')
    cl.clastering(level=0)
    x=cl.df['level_0'].value_counts()
    # n=list(x.index)
    # n0=n[0]
    # cl.clastering(level=1, claster={0:n0})
    # x=cl.df['level_1'].value_counts()
    # n=list(x.index)
    # n1=n[0]
    # cl.clastering(level=2, claster={0:n0, 1:n1})
    # x=cl.df['level_2'].value_counts()
    # n=list(x.index)
    # n2=n[0]
    
    # cl.clastering(level=3, claster={0:n0, 1:n1, 2:n2})
    # x=cl.df['level_3'].value_counts()
    # n=list(x.index)
    # print(x)
    # n3=n[0]
    
    # cl.clastering(level=4, claster={0:n0, 1:n1, 2:n2, 3:n3})
    # x=cl.df['level_4'].value_counts()
    # n=list(x.index)
    # print(x)
    # n4=n[0]
    
    # cl.clastering(level=5, claster={0:n0, 1:n1, 2:n2, 3:n3, 4:n4})
    # x=cl.df['level_5'].value_counts()
    # n=list(x.index)
    # print(x)
    # n5=n[0]
    # # cl.clastering(level=3, claster={0:0, 1:3, 2:n[0]})

    
    
    
            
        
        
    
    

        
        
        
        