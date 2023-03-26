#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    coco_network_neighbor_constraint.py test|actual
"""
from collections import defaultdict, Counter
from datetime import datetime
import sys
import numpy as np
import pandas as pd
import multiprocessing
import os
import random
import networkx as nx
import ujson as json
from utils import *
from jensen_shannon import *
from statistics import mean
from tqdm import tqdm
from ast import literal_eval

num_cores = 16
home_dir = "/zfs/projects/faculty/amirgo-identification/"
output_dir = os.path.join(home_dir, "coco_email_idtf_data/")

def generate_network_measures(timekey, edges, test_mode):
    """
    Generating network measures for a given time period using edges
    Parameters
    ----------
    timekey : str
        A string that represents the time period for which network measures are being computed
    edges : list
        A list of directd edges represented by 2-tuples
    test_mode : bool
        If true, restrict edges to a hundredth of all edges
    """
    if len(edges) < 10:
        sys.stderr.write('Returning empty network at %s with %d edges at %s.\n' % (timekey, len(edges), datetime.now()))
        return dict()

    G_directed = nx.DiGraph()
    sys.stderr.write('Generating weighted network measures for %s with %d edges at %s.\n' % (timekey, len(edges), datetime.now()))
    edges2weights = Counter(edges)
    weighted_edges = [(edge[0], edge[1], weight) for edge, weight in edges2weights.items()]
    G_directed.add_weighted_edges_from(weighted_edges)
    usr_quarter2network_measures = defaultdict(list)
    
    weighted_clustering = nx.clustering(G_directed, weight='weight')
    weighted_constraint = nx.constraint(G_directed, weight='weight')

    for n in G_directed:
        neighbor_clustering = [weighted_clustering[neighbor] for neighbor in G_directed.neighbors(n)]
        avg_clustering = (sum(neighbor_clustering) / len(neighbor_clustering)) if len(neighbor_clustering) > 0 else np.nan
        neighbor_constraint = [weighted_constraint[neighbor] for neighbor in G_directed.neighbors(n)]
        avg_constraint = (sum(neighbor_constraint) / len(neighbor_constraint)) if len(neighbor_constraint) > 0 else np.nan
        usr_quarter2network_measures[(n, timekey)] = [avg_clustering, avg_constraint]
    return dict(usr_quarter2network_measures)

def time_edges_to_df(time_edges, test_mode=False):
    """
    Calculates network measures using edge lists
    Parameters
    ----------
    time_edges : dict
        A dictionary that maps quarters (quarters only) to a list of edges belonging to that time period
    test_mode : bool, optional
        If true, only generate one network
    Returns
    -------
    df : pd.DataFrame
        A dataframe of network measures with user id and timekey_type as index
    """
    if test_mode:
        time_edges = {quarter:edges for quarter, edges in time_edges.items() if len(edges) > 5}
        test_timekey = random.choice(list(time_edges))
        sys.stderr.write("Testing timekey %s out of %d time periods.\n" % (test_timekey, len(time_edges)))
        network_measures = generate_network_measures(test_timekey, time_edges[test_timekey], test_mode)
    else:
        pool = multiprocessing.Pool(processes = num_cores)
        results = [pool.apply_async(generate_network_measures, args=(timekey, edges, test_mode, )) for timekey, edges in time_edges.items()]
        pool.close()
        pool.join()
        network_measures = defaultdict(list)
        for r in results:
            network_measures.update(r.get())
    
    cols = ['avg_alter_clustering', 'avg_alter_constraint']
    df = dict_to_df(network_measures, cols, index_name=['user_id', 'quarter'])
    return df.round(5)

def extract_network_measures(test_mode=False):
    """
    Main workhorse function for computing netwrork measures and writing them to file. Note that this function
    only computes measures quarterly.
    Parameters
    ----------
    test_mode : bool, optional
        If testing, modify file_name to include flags for testing in final output file name
    """
    edges_file = 'edges_test.txt' if test_mode else 'edges.txt'
    edges_file = os.path.join(output_dir, edges_file)

    quarterly_edges = defaultdict(list)
    sys.stderr.write("Reading edges from edge file at %s.\n" % str(datetime.now()))
    with open(edges_file, 'r') as f:
        for line in f:
            tup = literal_eval(line)
            quarterly_edges[tup[0]].append((tup[1], tup[2]))

    sys.stderr.write("Calculating network measures at %s.\n" % str(datetime.now()))
    file_name = 'coco_network_neighbor_constraint_test.csv' if test_mode else 'coco_network_neighbor_constraint.csv'
    
    df = time_edges_to_df(quarterly_edges, test_mode)
    df.to_csv(os.path.join(output_dir, file_name))
    
    sys.stderr.write("Finished outputting network measures at %s.\n" % str(datetime.now()))
    return

if __name__ == '__main__':
    starttime = datetime.now()
    test_mode = False
    try:
        test_mode = sys.argv[1].lower() == 'test'
    except IndexError as error:
        pass

    sys.stderr.write('Generating Network Measures at %s.\n' % datetime.now())
    extract_network_measures(test_mode)
    
    sys.stderr.write("Finished running at %s, with a duration of %s.\n"
        % (str(datetime.now()), str(datetime.now() - starttime)))
