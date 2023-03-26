#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    coco_union_bridging.py test|actual
"""
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing
from utils import *
import random
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

def generate_network_bridging(G, weighted):
    """
    Generate network bridging measures of G_directed
    Parameters
    ----------
    G : NetworkX graph
    weighted : bool
        Whether to engage in weighted computations
    edges2weights : dict of {tuple : int}
        Maps directed edges to weights
    Returns
    -------
    node2embed : dict
        Maps node to network bridging
    """
    node2embed = defaultdict(lambda : None)
    if weighted:
       for u in G:
            local_network = list(G.neighbors(u))
            within_cluster, without_cluster = 0, 0
            uv2weight = {}
            for v in local_network:
                uv2weight[v] = G.get_edge_data(u, v)['weight']
            total_weight = sum(uv2weight.values())
            for v in local_network:
                for w in G.neighbors(v):
                    add = (uv2weight[v]/total_weight) * G.get_edge_data(v, w)['weight']
                    if w in local_network:
                        within_cluster += add
                    else:
                        without_cluster += add
            if (without_cluster+within_cluster) == 0:
                node2embed[u] = np.nan
            else:
                # if 1, no within_cluster ties, if -1, all within-cluster ties
                node2embed[u] = (without_cluster-within_cluster)/(without_cluster+within_cluster)
    else:
       for u in G:
            local_network = list(G.neighbors(u))
            within_cluster, without_cluster = 0, 0
            for v in local_network:
                for w in G.neighbors(v):
                    if w in local_network:
                        within_cluster += 1
                    else:
                        without_cluster += 1
            if (without_cluster+within_cluster) == 0:
                node2embed[u] = np.nan
            else:
                # if 1, no within_cluster ties, if -1, all within-cluster ties
                node2embed[u] = (without_cluster-within_cluster)/(without_cluster+within_cluster)

    return node2embed

def generate_ego_bridging(G, weighted, node2community):
    """
    Generate the degree of ego bridging of G_directed. This is basically the EI index of the number of alters who are in ego's community to
    number of alters who are not in ego's community.
    Parameters
    ----------
    G_directed : NetworkX graph
    weighted : bool
        Whether to engage in weighted computations
    edges2weights : dict of {tuple : int}
        Maps directed edges to weights
    node2community : dict
        Maps nodes to the communities they belong to
    Returns
    -------
    node2embed : dict
        Maps node to ego bridging
    """
    node2embed = defaultdict(lambda : None)
    for u in G:
        local_community = node2community[u]
        within_cluster, without_cluster = 0, 0
        for v in G.neighbors(u):
            add = G.get_edge_data(u, v)['weight'] if weighted else 1
            if node2community[v] == local_community:
                within_cluster += add
            else:
                without_cluster += add
        if (without_cluster+within_cluster) == 0:
            node2embed[u] = np.nan
        else:
            node2embed[u] = (without_cluster-within_cluster)/(without_cluster+within_cluster)
    return node2embed

def generate_global_bridging(G, weighted, node2community):
    """
    Generate the degree of global bridging of G_directed. This is basically the EI index of the number of alters' alters who are in ego's community to
    number of alters who are not in ego's community.
    Parameters
    ----------
    G : NetworkX graph
    weighted : bool
        Whether to engage in weighted computations
    node2community : dict
        Maps nodes to the communities they belong to
    Returns
    -------
    node2embed : dict
        Maps node to global bridging
    """
    node2embed = defaultdict(lambda : None)
    if weighted:
        for u in G:
            local_community = node2community[u]
            within_cluster, without_cluster = 0, 0
            uv2weight = {}
            for v in G.neighbors(u):
                uv2weight[v] = G.get_edge_data(u, v)['weight']
            total_weight = sum(uv2weight.values())
            for v in G.neighbors(u):
                for w in G.neighbors(v):
                    add = (uv2weight[v]/total_weight) * G.get_edge_data(v, w)['weight']
                    if node2community[w] == local_community:
                        within_cluster += add
                    else:
                        without_cluster += add
            if (without_cluster+within_cluster) == 0:
                node2embed[u] = np.nan
            else:
                node2embed[u] = (without_cluster-within_cluster)/(without_cluster+within_cluster)
    else:
        for u in G:
            local_community = node2community[u]
            within_cluster, without_cluster = 0, 0
            for v in G.neighbors(u):
                for w in G.neighbors(v):
                    if node2community[w] == local_community:
                        within_cluster += 1
                    else:
                        without_cluster += 1
            if (without_cluster+within_cluster) == 0:
                node2embed[u] = np.nan
            else:
                node2embed[u] = (without_cluster-within_cluster)/(without_cluster+within_cluster)

    
    return node2embed

def generate_community_ei(G_directed, G, weighted):
    """
    Generate community-based bridging measures by identifying the community structure and calling relevant functions.
    Parameters
    ----------
    G_directed : nx.DiGraph()
        This the original version of the graph
    G : nx.Graph()
        This is the verison of the graph that only retains reciprocal edges
    weighted : bool
        Whether to engage in weighted computations
    Returns
    -------
    (node2ego_bridging, node2community_bridging, len(communities)) : tuple of (dict, dict, int)
    """
    if weighted:
        communities = algorithms.leiden(G_directed, weights='weight').communities
    else:
        communities = algorithms.leiden(G_directed).communities
    node2community = {node : i for i, c in enumerate(communities) for node in c}

    node2ego_bridging = generate_ego_bridging(G, weighted, node2community)
    node2community_bridging = generate_global_bridging(G, weighted, node2community)

    return (node2ego_bridging, node2community_bridging, len(communities))

def find_union_graph(G_directed):
    """
    Find a graph that retains all edges and removes directions so that each node's neighbors are both incoming neighbors and outgoing neighbors
    Parameters
    ----------
    G_directed : nx.DiGraph()
    Returns
    -------
    G : nx.Graph()
        The undirected version of G_directed that retains all incoming and outgoing edges
    """
    G = nx.Graph()
    for u, v, weight in G_directed.edges(data=True):
        if not G.has_edge(u, v):
            new_weight = 0
            if G_directed.has_edge(v, u):
                new_weight = (weight['weight'] + G_directed.get_edge_data(v, u)['weight'])/2
            else:
                new_weight = weight['weight'] / 2
            G.add_edge(u, v, weight=new_weight)
    return G

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
    
    G = find_union_graph(G_directed)
    usr_quarter2network_measures = defaultdict(list)
    # embeddedness, connectivity, and bridging are different versions of the same variable
    sys.stderr.write('Computing bridging measures at %s.\n' % datetime.now())
    unweighted_node2network_bridging = generate_network_bridging(G, False)
    weighted_node2network_bridging = generate_network_bridging(G, True)

    unweighted_node2ego_bridging, unweighted_node2community_bridging, unweighted_n_comm = generate_community_ei(G_directed, G, False)
    weighted_node2ego_bridging, weighted_node2community_bridging, weighted_n_comm = generate_community_ei(G_directed, G, True)
    
    for n in G:
        usr_quarter2network_measures[(n, timekey)] = [unweighted_node2network_bridging[n], unweighted_node2ego_bridging[n], unweighted_node2community_bridging[n], unweighted_n_comm,
        weighted_node2network_bridging[n], weighted_node2ego_bridging[n], weighted_node2community_bridging[n], weighted_n_comm]
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
    
    cols = (['network_bridging_unweighted', 'ego_bridging_unweighted', 'global_bridging_unweighted', 'n_comm_unweighted', 'network_bridging_weighted', 'ego_bridging_weighted', 'global_bridging_weighted', 'n_comm_weighted'])
    
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
    file_name = 'coco_union_bridging_test.csv' if test_mode else 'coco_union_bridging.csv'
    
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
