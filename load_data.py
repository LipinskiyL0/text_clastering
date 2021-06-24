# -*- coding: utf-8 -*-
"""
производим обработку текста в рамках социологического исследования

"""

from keras.datasets import mnist
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, ward, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from morph_anlysis_me import morph_anlysis_me

import pickle

def clear_text(s, musor):
    for m in musor:
        if s==m:
            return False
    return True
#Загружаем 92 опрос    
df92=pd.read_csv('rez_table92.csv')
# d2=pd.read_excel('code_table2.xlsx')

#обрабатываем 91 опрос
df91=pd.read_excel('rez_table.xlsx')

#выравниваем файл
df91['syllabus_caption'] = df91['syllabus_caption'].fillna(method='ffill')
df91['study_group_caption'] = df91['study_group_caption'].fillna(method='ffill')
df91['opros']=['91']*len(df91)    

cols=['topic_poll_id', 'syllabus_caption', 'study_group_caption', 'user_id',
       'finished_at',]
cols=['Q18',	'Q19',	'Q20',	'Q21',	'Q22',	'Q23']

df911=df91[cols]
df921=df92[cols]

df=pd.concat([df911, df921], axis=0)
df=df.reset_index()
del df['index']


musor=['0.0','0',0,1, '-', '.' , '0', '1', '...', '*', ',','_', '?', '!!!', '..', '---', 
       'а', '!', 'В', 'СССССС', 'бббббббббббббббббббб', 'Н', ]



for c in cols:
    ind_Text=df[c].apply(clear_text, args=(musor,))
    Text=df.loc[ind_Text, c]
    Text=morph_anlysis_me(Text)
    df.loc[ind_Text, c+'_txt']=Text
    print(c)

with open('rez_table_text.pkl', 'wb') as f:
     pickle.dump(df, f)


