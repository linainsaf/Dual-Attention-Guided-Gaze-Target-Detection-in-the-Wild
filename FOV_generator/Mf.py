# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 14:08:42 2021

@author: Admin
"""

import numpy as np

def Mf(H, G, I, alpha=6):
    """
        H     : (hx, hy) head position
        G     : (gx, gy) gaze target direction
        I     : image
        alpha : parameter to decide the angle of view
        
        return : FOV attention Map
    """
    
    FOV = np.zeros_like(I)
    
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            
            theta   = np.arccos( ( (np.array([i,j])-H)*G )/( np.linalg.norm(np.array([i,j])-H)*np.linalg.norm(G)) )
            FOV[i,j] = np.max(1-((alpha*theta)/np.pi), 0)
    
    return FOV