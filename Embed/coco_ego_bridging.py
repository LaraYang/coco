#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    coco_network_supplementary.py test|actual unweighted|weighted|both
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
import networkx.algorithms.community as nx_comm
from cdlib import algorithms

num_cores = 16
build_threshold_network = False
home_dir = "/zfs/projects/faculty/amirgo-identification/"
email_dir = os.path.join(home_dir, "email_data/")
email_file = os.path.join(email_dir, 'MessagesHashed.jsonl')
users_file = os.path.join(email_dir, 'Users.json')
activity_file = os.path.join(email_dir, 'Activities.json')
output_dir = os.path.join(home_dir, "coco_email_idtf_data/")
user_qualtrics_file = os.path.join(home_dir, "survey_hr_data", "UsersQualtrics.csv")
long_hr_file = "/zfs/projects/faculty/amirgo-transfer/spacespace/spacespace/Coco/analyses_data/longitudinal_hr.csv"

email2user_id = {}

domain_hash = {
    'collabera.com':                     '509c8f6b1127bceefd418c023533d653', 
    'collaberainc.mail.onmicrosoft.com': 'ec5b67548b6ec06f1234d198efec741e', 
    'collaberainc.onmicrosoft.com':      '86160680578ee9258f097a67a5f25af9', 
    'collaberasp.com':                   '6bf3934d19f1acf5b9295b63e0e7f66e', 
    'g-c-i.com':                         '3444d1f7d5e46443080f2d069e41a10c'}
collabera_hashes = set([v for k, v in domain_hash.items()])

lines_to_test = 2000

def generate_ego_community_ei(G_directed, edges2weights, weighted):
    if weighted:
        communities = algorithms.leiden(G_directed, weights='weight').communities
    else:
        communities = algorithms.leiden(G_directed).communities
    node2community = {node : i for i, c in enumerate(communities) for node in c}
    node2embed = defaultdict(lambda : None)
    for u in G_directed:
        local_community = node2community[u]
        within_cluster, without_cluster = 0, 0
        for v in G_directed.neighbors(u):
            if node2community[v] == local_community:
                within_cluster += 1
            else:
                without_cluster += 1

        if (without_cluster+within_cluster) == 0:
            node2embed[u] = np.nan
        else:
            node2embed[u] = (without_cluster-within_cluster)/(without_cluster+within_cluster)
    return (node2embed, len(communities))

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

    sys.stderr.write('Computing community-based connectivity at %s.\n' % datetime.now())
    node2embed_weighted, node2embed_unweighted = dict(), dict()
    n_comm_weighted, n_comm_unweighted = 0, 0
    
    node2embed_unweighted, n_comm_unweighted = generate_ego_community_ei(G_directed, edges2weights, False)
    node2embed_weighted, n_comm_weighted = generate_ego_community_ei(G_directed, edges2weights, True)
    

    for n in G_directed:
        row = [node2embed_unweighted[n], n_comm_unweighted, node2embed_weighted[n], n_comm_weighted]
        usr_quarter2network_measures[(n, timekey)] = row
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
    
    cols = ['ego_bridging_unweighted', 'n_comm_unweighted', 'ego_bridging_weighted', 'n_comm_weighted']
    
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
    file_name = 'coco_ego_bridging_test.csv' if test_mode else 'coco_ego_bridging.csv'
    
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
