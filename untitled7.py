# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:37:05 2021

@author: Leonid
"""

def cartesian_iterative(pools):
    result = [[]]
    for pool in pools:
      result = [x+[y] for x in result for y in pool]
    return result

xx=cartesian_iterative([[1,2,3],['a','b'],[4,5]])
print(xx)