# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:24:51 2021

@author: Leonid
"""
import pickle

from sklearn.pipeline import Pipeline
from my_clasterizator2 import KMeansClusters2
from my_vectorizer import vectorizer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from my_split import my_cv_spliter

def get_outputs(X):
    #Для многократной кластеризации отделяем во входной выборке входы от меток
    cols=list(X.columns)
    cols_out=[x for x in cols if ('level' in x)==True]
    return cols_out

def get_samples(df, n_str=10):
    flag=0
    cols_out=get_outputs(df)
    for col in reversed(cols_out):
        ind=df[col].notnull()
        df1=df[ind]
        xx=list(df1[col].value_counts().index)
        xx.sort()
        
        for x in xx:
            df2=df1[df1[col]==x].sample(n=n_str,  replace=True)
            if flag==0:
                df_rez=df2.copy()
                flag=1
            else:
                df_rez=pd.concat([df_rez, df2], axis=0)
    return df_rez
        
    

if __name__ == '__main__':
    
    
    with open('rez_table_text.pkl', 'rb') as f:
        df  = pickle.load(f)
    name='Q20'
    ind=df[name+'_txt'].notnull()
    r=df[name].loc[ind]
    Result=list(df[name+'_txt'].loc[ind]) 
    pipe = Pipeline([("vect", vectorizer(method='tfidf', min_df=0.03, max_df=0.9, ngram_range=(1,1) )),
                      ("c1", KMeansClusters2(n_clusters=10, f_opt=False, max_iter=300, n_init=100)),
                        # ("c2", KMeansClusters2(n_clusters=5)),
                        
                        # ("c3", KMeansClusters2(n_clusters=3)),
                          # ("c4", KMeansClusters2(n_clusters=3)), 
                          # ("c5", KMeansClusters2(n_clusters=3)),
                          # ("c6", KMeansClusters2(n_clusters=3)),
                          # ("c7", KMeansClusters2(n_clusters=3)),
                          # ("c8", KMeansClusters2(n_clusters=3))
                     ])
    
    param_grid = {'vect__min_df': [0.01,   0.02, 0.03 ],
                  'vect__max_df': [0.85, 0.9, 0.95],
                  'vect__method':['tfidf', 'bw'],
                   # 'vect__LDA_n_components':[10, 30, 50,  70, 90],
                   #  'vect__LDA_n_lexems':[10, 30, 50,  70, 90],
                  'c1__n_clusters': [5, 7,10,  15, 20, 25, 30]
                   }
    grid = GridSearchCV(pipe, param_grid, return_train_score=True, cv=my_cv_spliter )
   
    grid=grid.fit(Result)
    print(grid.best_params_)
    X11=grid.predict(Result)
    rez=grid.score(Result)
    pipe=grid.best_estimator_
    # pipe=pipe.fit(Result)
    X11=pipe.predict(Result)
    rez=pipe.score(Result)
    centroids=pipe[-1].get_centroid_all_ind(X11)
    X11['sample']=0
    X11.loc[centroids, 'sample']=1
    
    cols_out=get_outputs(X11)
    if 'sample'in list(X11.columns):
        cols_out1=cols_out+['sample']
        
    df1=df.loc[ind, [name, name+'_txt']]
    df1=df1.reset_index()
    del df1['index']
    
    df1=pd.concat([df1, X11[cols_out1]], axis=1, ignore_index=False)
    # xx=df1.iloc[:, -1].value_counts()
    df2=df1.fillna(-1)
    xx=df2.groupby(cols_out)[name].count()
    xx=xx.reset_index()
    if 'sample' in xx.columns:
        del xx['sample']
    xx=xx.rename(columns={name:'num'})
    print(xx)
    print(pipe[-1].silhouette_avg)
    # print(grid.best_params_)
    df2= get_samples(df1, n_str=20)
    df2=pd.merge(df2, xx, how='left', left_on=cols_out, right_on=cols_out)
    
    if 'sample' in df1.columns:
        del df1['sample']
        
    if 'sample' in df2.columns:
        del df2['sample']
    
    df1.to_excel(name+'_all.xlsx')
    df2.to_excel(name+'_samples_random.xlsx')
    xx.to_excel(name+'_freq.xlsx')
    df3= pipe[-1].get_samples_ind(X11, n=20)
    df3=df1.loc[df3, :]
    if 'sample' in df3.columns:
        del df3['sample']
    df3=df3.fillna(-1)
    
    
    df3=pd.merge(df3, xx, how='left', left_on=cols_out, right_on=cols_out)
    df3.to_excel(name+'_samples_silhouette.xlsx')
    
    
    
    
    