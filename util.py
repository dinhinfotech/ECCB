# -*- coding: utf-8 -*-
"""
Util file includes utility functions
"""
from os import listdir
from os.path import isfile, join
import numpy as np


def list_files_in_folder(folder_path):    
    """
    Return: A list of the file names in the folder
    """
          
    list = listdir(folder_path)
    onlyfiles = [ f for f in list  if isfile(join(folder_path,f)) ]
    return onlyfiles 

def load_list_from_file(file_path):
    """
    Return: A list saved in a file
    """
    
    f = open(file_path,'r')
    listlines = [line.rstrip() for line in f.readlines()]
    f.close()
    return listlines

def load_matrices(folder_path):
    """
    Return: A list of matrices saved in the folder
    """
    
    file_names = list_files_in_folder(folder_path)   
    matrices = []
    
    for file_name in file_names:
        matrices.append(np.loadtxt(folder_path + file_name))
        
    return matrices  

def extract_submatrix(row_indices, col_indices, A):
    """ Extracting a submatrix from  matrix A
    
    Parameter:
    row_indices: row index list that we want to extract
    col_indices: Column index list that we want to extract
    A: Matrix
    
    Return:
    submatrix of A
    """

    len_row = len(row_indices)
    len_col = len(col_indices)
    
    #M = matrix(0.0,(len_row,len_col))
    M = np.zeros((len_row,len_col))
    for order1, idx_row in enumerate(row_indices):
        for order2, idx_col in enumerate(col_indices):
            M[order1,order2] = A[idx_row,idx_col]
    
    return M 