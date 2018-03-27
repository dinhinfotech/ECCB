# -*- coding: utf-8 -*-
from eden.graph import Vectorizer
from sklearn.metrics import pairwise

class CDNK_Vectorizer():
    def __init__(self,
                 r=1,
                 d=2,
                 nbits=20,
                 discrete=True,
                 n_jobs=1):
        """ Constructor
        
        Parameters:
            - max_deg: 
            - cli_threshold:
            - r: r
            - d: distance
            - nbits:
            - n_jobs:
                
        """                     
        self.r = r
        self.d = d
        self.nbits = nbits
        self.discrete=discrete

    def vectorize(self, g):
        """ Vectorize graph nodes
        
        Return: matrix in which rows are the vectors that represents for nodes        
        """
        
        vec = Vectorizer(nbits=self.nbits, 
                         discrete=self.discrete, 
                         r=self.r, 
                         d=self.d)
                         
        M = vec.vertex_transform([g])[0]  
                     
        return M

    def cdnk(self, g):
        """Compute graph node kernel matrix encoding node similarities
        
           Return: 
           Kernel matrix
        """            
        M = self.vectorize(g)
        K = pairwise.linear_kernel(M,M)
        
        return K        