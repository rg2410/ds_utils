# -*- coding: utf-8 -*-
"""
Data Science Utils

Find correlation between predictors
Author: Rafael Fernandes
"""

import operator

import numpy as np
import pandas as pd

import networkx as nx


class CorrAnalysis:
    
    def __init__(self, data):
        self.data = data
        self.graph = self.create_graph()
        self.original_graph = self.graph.copy()
    

    def create_graph(self):
        df = self.data.corr().stack().reset_index()
        df = df[df['level_0']!=df['level_1']]
        df.columns = ['from', 'to', 'weight']
        df['sign'] = df.weight.apply(np.sign)
        df.weight = df.weight.abs()
        return nx.from_pandas_dataframe(df, 'from', 'to', 
                                        edge_attr=['weight', 'sign'])


    def filter_graph(self, thresh):
        self.graph = nx.Graph(((source, target, attr) for \
                           source, target, attr in \
                           self.graph.edges_iter(data=True) \
                           if attr['weight'] > thresh))
    

    def clear_filter(self):
        self.graph = self.original_graph


    def get_corr_features(self, thresh):
        self.delete_list = []
        self.filter_graph(thresh)
        temp = self.graph.copy()
        while temp.nodes():
            node = max(temp.degree(weight='weight').items(), key=operator.itemgetter(1))[0]
            to_delete = temp.neighbors(node)
            self.delete_list.extend(to_delete)
            to_delete.append(node)
            temp.remove_nodes_from(to_delete)
        return self.delete_list
    
    
    def plot_hist(self):
        data = [i[2] for i in self.graph.edges_iter(data='weight')]
        pd.Series(data).hist()