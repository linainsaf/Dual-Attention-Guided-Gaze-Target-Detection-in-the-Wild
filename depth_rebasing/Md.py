# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 18:24:27 2021

@author: Admin
"""

import numpy as np

def Md (Id,N,gz):
    
    r=16  # constante
    sigma=0.3 #constante
       
    Fd=Id-np.sum(np.sum(N))/N.size
    
    
    Mfront = np.maximum(Fd, np.zeros_like(Fd))
    Mmid = np.maximum(1-r*Fd**2, np.zeros_like(Fd))
    Mback = np.maximum(-Fd, np.zeros_like(Fd))
    
    if gz >-1 or gz<sigma:
        Md = Mfront
    elif gz >-sigma or gz<sigma:
        Md = Mmid
    elif gz >sigma or gz<1:
        Md = Mback
    
    return Md
    