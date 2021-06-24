# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:24:51 2021

@author: Leonid
"""

from sklearn.pipeline import Pipeline
from my_clasterizator2 import KMeansClusters2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    X1=np.random.randn(100, 1)
    X2=np.random.randn(100, 1)
    
    Y1=np.random.randn(100, 1)+5
    Y2=np.random.randn(100, 1)+5
    X=np.concatenate([X1, X2], axis=1)    
    Y=np.concatenate([Y1, Y2], axis=1)
    
    X=np.concatenate([X, Y], axis=0)
    X=pd.DataFrame(X, columns=['t1', 't2'])

    pipe = Pipeline([("c1", KMeansClusters2(n_clusters=4)), ("c2", KMeansClusters2(n_clusters=3)),
                     ("c3", KMeansClusters2(n_clusters=2))])
    pipe=pipe.fit(X)
    X11=pipe.predict(X)
    rez=pipe.score(X)
    
