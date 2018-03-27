from graph import Vectorization
import util
import graph_util as gu
from sklearn import cross_validation
from collections import defaultdict
from sklearn import svm
from sklearn import metrics
import networkx as nx

class Validation:
    def __init__(self,
                 kernels=None,
                 training_genes=None,
                 training_labels=None,
                 list_all_genes=None,
                 dict_gene_idx=None,
                 list_r=None,
                 list_d=None, 
                 list_c=None,
                 n_folds=3
                 ):
        self.kernels = kernels
        self.training_genes = training_genes
        self.training_labels = training_labels
        self.list_all_genes = list_all_genes
        self.dict_gene_idx = dict_gene_idx
        self.list_r = list_r
        self.list_d = list_d
        self.list_c = list_c
        self.n_folds = n_folds
        self.paras = None
        
    def para_selection(self, validation_genes=None, validation_labels=None):        
        
        validation_idx = [self.dict_gene_idx[gene] for gene in validation_genes]    
        n_validation_genes = len(validation_genes)
        
        dict_para_auc = defaultdict(lambda: 0)
        
        kf = cross_validation.KFold(n_training_genes, n_folds = self.n_folds)
        
        for train_index, test_index in kf:
            
            for dec_para in dec_paras:
                vec = Vectorization(max_deg=dec_para[0], cli_threshold=dec_para[1],
                                    max_node_ID=self.max_node_ID)                           
                g_dec = vec.decompose(g_sub)
                                
                dict_nodeidx_rowidx = {}
                for row_idx, node_idx in enumerate(g_dec.nodes()):
                    dict_nodeidx_rowidx[node_idx] = row_idx
                    
                sub_graph_row_index = [dict_nodeidx_rowidx[node_idx] for node_idx in sub_node_idx]                    
                
                validation_idx = [training_idx[idx] for idx in train_index]
                validation_row_idx = [dict_nodeidx_rowidx[idx] for idx in validation_idx]                 
                validation_labels = [training_labels[idx] for idx in train_index]
                
                test_idx = [training_idx[idx] for idx in test_index]
                test_row_idx = [dict_nodeidx_rowidx[idx] for idx in test_idx]                                                                
                test_labels = [training_labels[idx] for idx in test_index]
                
                unknown_row_idx = test_row_idx[:]
                for index in range(len(g_dec.nodes())):
                    if index not in validation_row_idx and index not in test_row_idx and\
                                index in sub_graph_row_index:
                        unknown_row_idx.append(index)                                        
                    
                    M = vec.vectorize(g_dec)
                    
                    M_val = M[validation_row_idx,:]
                    
                    M_unknown = M[unknown_row_idx,:]
                    
                    for c in self.list_c:                        
                        clf = svm.LinearSVC(C=c)
                        clf.fit(M_val, validation_labels)
                        scores = clf.decision_function(M_unknown)
                        
                        qscores = []
                        
                        for s in scores[:len(test_idx)]:
                            qscore = float(sum([int(s >= value) for value in scores]))/len(scores)
                            qscores.append(qscore)
                        fpr, tpr, thresholds = metrics.roc_curve(test_labels, qscores, pos_label= 1)
                        auc = metrics.auc(fpr, tpr)
                        dict_para_auc[(dec_para,vec_para,c)]+=auc

        optimal_paras = max(dict_para_auc.iterkeys(), key=lambda k: dict_para_auc[k])
        self.paras = optimal_paras
        #print "AUC: ", dict_para_auc[optimal_paras]
        return optimal_paras

    def evaluation(self, optimal_paras=None):
        
        g = gu.load_graph(self.adjacency_matrix_path)
        g_sub = gu.get_subgraph(g,self.list_all_genes_path, self.list_training_genes_path)
        sub_node_idx = g_sub.nodes()
                
        vec = Vectorization(max_deg = optimal_paras[0][0],
                   cli_threshold = optimal_paras[0][1],
                   r = optimal_paras[1][0],
                   d = optimal_paras[1][1],
                   max_node_ID=self.max_node_ID)
        
        g_dec = vec.decompose(g_sub)
        
        dict_nodeidx_rowidx = {}
        for row_idx, node_idx in enumerate(g_dec.nodes()):
            dict_nodeidx_rowidx[node_idx] = row_idx
            
        sub_graph_row_index = [dict_nodeidx_rowidx[node_idx] for node_idx in sub_node_idx]
            
        list_all_genes = util.load_list_from_file(self.list_all_genes_path)        
        dict_gene_idx = {}
        for idx, gene in enumerate(list_all_genes):
            dict_gene_idx[gene]=idx
        
        training_genes = util.load_list_from_file(self.list_training_genes_path)
        training_idx = [dict_gene_idx[gene] for gene in training_genes]
        training_row_idx = [dict_nodeidx_rowidx[idx] for idx in training_idx]
        training_labels = [int(e) for e in util.load_list_from_file(self.list_training_labels_path)]        
        
        #"""Labeling"""
        #dict_node_attvalues = {}
        #for n in g_dec.nodes_iter():
        #        dict_node_attvalues[n]= ""

        #nx.set_node_attributes(g_dec,'label',dict_node_attvalues)          
        
        M = vec.vectorize(g_dec)
        qscores = []
        
        for idx, row_idx in enumerate(training_row_idx):
            validation_row_idx = training_row_idx[:]
            del validation_row_idx[idx]
            
            validation_labels = training_labels[:]
            del validation_labels[idx]
            
            unknown_row_idx = [row_idx]
            for index in range(M.shape[0]):
                if (index not in training_row_idx) and (index in sub_graph_row_index):
                    unknown_row_idx.append(index)

            M_val = M[validation_row_idx,:]            
            
            M_unknown = M[unknown_row_idx,:]
            
            
            clf = svm.LinearSVC(C=optimal_paras[2])
            clf.fit(M_val, validation_labels)
            scores = clf.decision_function(M_unknown)
                        
            qscore = float(sum([int(scores[0] >= val) for val in scores]))/(len(scores)-1)
            
            qscores.append(qscore)
            
            
        fpr, tpr, thresholds = metrics.roc_curve(training_labels, qscores, pos_label= 1)
        auc = metrics.auc(fpr, tpr)
        
        return auc

    def evaluation_biogridphys(self, optimal_paras=None):
        
        g = gu.load_graph(self.adjacency_matrix_path)
        g_sub = gu.get_subgraph(g,self.list_all_genes_path, self.list_training_genes_path)
        sub_node_idx = g_sub.nodes()
                
        vec = Vectorization(max_deg = optimal_paras[0][0],
                   cli_threshold = optimal_paras[0][1],
                   r = optimal_paras[1][0],
                   d = optimal_paras[1][1],
                   max_node_ID=self.max_node_ID)
        
        g_dec = vec.decompose_biogridphys(g_sub)
        
        dict_nodeidx_rowidx = {}
        for row_idx, node_idx in enumerate(g_dec.nodes()):
            dict_nodeidx_rowidx[node_idx] = row_idx
            
        sub_graph_row_index = [dict_nodeidx_rowidx[node_idx] for node_idx in sub_node_idx]
            
        list_all_genes = util.load_list_from_file(self.list_all_genes_path)        
        dict_gene_idx = {}
        for idx, gene in enumerate(list_all_genes):
            dict_gene_idx[gene]=idx
        
        training_genes = util.load_list_from_file(self.list_training_genes_path)
        training_idx = [dict_gene_idx[gene] for gene in training_genes]
        training_row_idx = [dict_nodeidx_rowidx[idx] for idx in training_idx]
        training_labels = [int(e) for e in util.load_list_from_file(self.list_training_labels_path)]        
        
        #"""Labeling"""
        #dict_node_attvalues = {}
        #for n in g_dec.nodes_iter():
        #        dict_node_attvalues[n]= ""

        #nx.set_node_attributes(g_dec,'label',dict_node_attvalues)          
        
        M = vec.vectorize(g_dec)
        qscores = []
        
        for idx, row_idx in enumerate(training_row_idx):
            validation_row_idx = training_row_idx[:]
            del validation_row_idx[idx]
            
            validation_labels = training_labels[:]
            del validation_labels[idx]
            
            unknown_row_idx = [row_idx]
            for index in range(M.shape[0]):
                if (index not in training_row_idx) and (index in sub_graph_row_index):
                    unknown_row_idx.append(index)

            M_val = M[validation_row_idx,:]            
            
            M_unknown = M[unknown_row_idx,:]
            
            
            clf = svm.LinearSVC(C=optimal_paras[2])
            clf.fit(M_val, validation_labels)
            scores = clf.decision_function(M_unknown)
                        
            qscore = float(sum([int(scores[0] >= val) for val in scores]))/(len(scores)-1)
            
            qscores.append(qscore)
            
            
        fpr, tpr, thresholds = metrics.roc_curve(training_labels, qscores, pos_label= 1)
        auc = metrics.auc(fpr, tpr)
        
        return auc        