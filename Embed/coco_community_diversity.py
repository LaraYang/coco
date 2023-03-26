#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    coco_community_diversity.py test|actual
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
home_dir = "/zfs/projects/faculty/amirgo-identification/"
email_dir = os.path.join(home_dir, "email_data/")
email_file = os.path.join(email_dir, 'MessagesHashed.jsonl')
users_file = os.path.join(email_dir, 'Users.json')
activity_file = os.path.join(email_dir, 'Activities.json')
output_dir = os.path.join(home_dir, "coco_email_idtf_data/")

survey_filename = "/ifs/gsb/amirgo/spacespace/spacespace/Coco/analyses_data/preprocessed_survey_hr.csv"
user_qualtrics_file = os.path.join(home_dir, "survey_hr_data", "UsersQualtrics.csv")
hr_file = os.path.join(home_dir, "survey_hr_data", "Collabera_HR_Perf.csv")
long_hr_file = "/zfs/projects/faculty/amirgo-transfer/spacespace/spacespace/Coco/analyses_data/longitudinal_hr.csv"

email2user_id, quarter2usr2department = {}, defaultdict(dict)

def fill_demographics():
    """
    This version is slightly different than older versions of fill_demographic as we are only interested in 
    computing department-based EI here. In addition, email2user_id needs to be filled first as we are no longer
    reading edges from scratch in this method.
    """
    global email2user_id,user_id2department
    with open(users_file, encoding='utf-8') as f:
        for line in f:
            user = json.loads(line)
            for e in user['Emails']:
                email2user_id[e] = user['UserId']

    user_qualtrics_df = pd.read_csv(user_qualtrics_file)[['Email', 'UID']]
    user_qualtrics_df.set_index('UID', inplace=True)
    uid2email = user_qualtrics_df['Email'].to_dict()
    
    hr_df_quarterly = pd.read_csv(long_hr_file)
    hr_df_quarterly['user_id'] = hr_df_quarterly.apply(lambda row : email2user_id.get(uid2email.get(row['uid'])), axis=1)
    hr_df_quarterly.set_index(['user_id', 'quarter'], inplace=True)
    
    global quarter2usr2department
    userid_quarter2department = hr_df_quarterly['department'].dropna().apply(lambda val : val.strip().lower()).to_dict()
    for k, department in userid_quarter2department.items():
        u, quarter = k
        quarter2usr2department[quarter][u] = department

    return

def find_all_neighbors(G_directed, u):
    """
    Find the union of in and out-neighbors of u in G_directed.
    """
    return set(G_directed.successors(u)).union(set(G_directed.predecessors(u)))

def generate_department_bridging(G_directed, usr2department, edges2weights, weighted):
    """
    Generate department-based EI index
    Parameter
    ---------
    G_directed : NetworkX graph
    usr2department : dict
        Maps useres to departments
    edges2weights: dict
        Maps directed weights to weights
    weighted : bool
        Whether computations should be weighted
    Returns
    -------
    tuple
        A 2-tuple of dictionaries matching node (i.e., user ids) to department EI index using two kinds of neighbors.
    """
    node2embed, node2embed_all_neighbors = defaultdict(lambda : None), defaultdict(lambda : None)
    for u in G_directed:    
        if u in usr2department:
            within_department, without_department = 0, 0
            for v in G_directed.neighbors(u):
                if v in usr2department:
                    add = edges2weights[(u, v)] if weighted else 1
                    if usr2department[u] == usr2department[v]:
                        within_department += add
                    else:
                        without_department += add
            if (within_department + without_department == 0):
                node2embed[u] = np.nan
            else:
                node2embed[u] = ((without_department - within_department) / (without_department + within_department))
            
            within_department, without_department = 0, 0
            for v in find_all_neighbors(G_directed, u):
                if v in usr2department:
                    add = (edges2weights.get((u, v), 0) + edges2weights.get((v, u), 0)) if weighted else 1
                    if usr2department[u] == usr2department[v]:
                        within_department += add
                    else:
                        without_department += add
            if (within_department + without_department == 0):
                node2embed_all_neighbors[u] = np.nan
            else:
                node2embed_all_neighbors[u] = ((without_department - within_department) / (without_department + within_department))

    return (node2embed, node2embed_all_neighbors)


def generate_ego_bridging(G_directed, edges2weights, communities, weighted):
    """
    Generating community EI based on immediate alters. This function returns both the results of using successors only vs
    using both successors and predecessor nodes.
    The unweighted version of both successor only and successor and predecessor both measure have already been separately computed in
    corpcorp_ego_bridging.py and corpcorp_union_bridging.py. They are re-computed here so that there's a correctly computed weighted version
    of these measures.
    """
    node2embed, node2embed_all_neighbors = defaultdict(lambda : None), defaultdict(lambda : None)
    node2community = {node : i for i, c in enumerate(communities) for node in c}

    for u in G_directed:
        local_community = node2community[u]
        within_cluster, without_cluster = 0, 0
        for v in G_directed.neighbors(u):
            add = edges2weights[(u, v)] if weighted else 1
            if node2community[v] == local_community:
                within_cluster += add
            else:
                without_cluster += add

        if (without_cluster+within_cluster) == 0:
            node2embed[u] = np.nan
        else:
            node2embed[u] = (without_cluster-within_cluster)/(without_cluster+within_cluster)

        within_cluster, without_cluster = 0, 0
        for v in find_all_neighbors(G_directed, u):
            add = (edges2weights.get((u, v), 0) + edges2weights.get((v, u), 0)) if weighted else 1
            if node2community[v] == local_community:
                within_cluster += add
            else:
                without_cluster += add

        if (without_cluster+within_cluster) == 0:
            node2embed_all_neighbors[u] = np.nan
        else:
            node2embed_all_neighbors[u] = (without_cluster-within_cluster)/(without_cluster+within_cluster)
    
    return (node2embed, node2embed_all_neighbors)

def generate_global_bridging(G_directed, edges2weights, communities, weighted):
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
    node2embed, node2embed_all_neighbors = defaultdict(lambda : None), defaultdict(lambda : None)
    node2community = {node : i for i, c in enumerate(communities) for node in c}

    if weighted:
        for u in G_directed:
            local_community = node2community[u]
            within_cluster, without_cluster = 0, 0
            uv2weight = {v : edges2weights[(u, v)] for v in G_directed.neighbors(u)}
            total_weight = sum(uv2weight.values())
            for v in G_directed.neighbors(u):
                for w in G_directed.neighbors(v):
                    add = (uv2weight[v]/total_weight) * edges2weights[(v, w)]
                    if u == w:
                        continue
                    elif node2community[w] == local_community:
                        within_cluster += add
                    else:
                        without_cluster += add
            if (without_cluster+within_cluster) == 0:
                node2embed[u] = np.nan
            else:
                node2embed[u] = (without_cluster-within_cluster)/(without_cluster+within_cluster)

            all_neighbors = find_all_neighbors(G_directed, u)
            uv2weight = {v : (edges2weights.get((u, v), 0) + edges2weights.get((v, u), 0)) for v in all_neighbors}
            total_weight = sum(uv2weight.values())
            for v in all_neighbors:
                for w in find_all_neighbors(G_directed, v):
                    add = (uv2weight[v]/total_weight) * (edges2weights.get((v, w), 0) + edges2weights.get((w, v), 0))
                    if u == w:
                        continue
                    elif node2community[w] == local_community:
                        within_cluster += add
                    else:
                        without_cluster += add
            if (without_cluster+within_cluster) == 0:
                node2embed_all_neighbors[u] = np.nan
            else:
                node2embed_all_neighbors[u] = (without_cluster-within_cluster)/(without_cluster+within_cluster)
    else:
        for u in G_directed:
            local_community = node2community[u]
            within_cluster, without_cluster = 0, 0
            for v in G_directed.neighbors(u):
                for w in G_directed.neighbors(v):
                    if u == w:
                        continue
                    elif node2community[w] == local_community:
                        within_cluster += 1
                    else:
                        without_cluster += 1
            if (without_cluster+within_cluster) == 0:
                node2embed[u] = np.nan
            else:
                node2embed[u] = (without_cluster-within_cluster)/(without_cluster+within_cluster)

            within_cluster, without_cluster = 0, 0
            for v in find_all_neighbors(G_directed, u):
                for w in find_all_neighbors(G_directed, v):
                    if u == w:
                        continue
                    elif node2community[w] == local_community:
                        within_cluster += 1
                    else:
                        without_cluster += 1
            if (without_cluster+within_cluster) == 0:
                node2embed_all_neighbors[u] = np.nan
            else:
                node2embed_all_neighbors[u] = (without_cluster-within_cluster)/(without_cluster+within_cluster)
    return (node2embed, node2embed_all_neighbors)

def generate_community_diversity(G_directed, edges2weights, communities, weighted):
    """
    Computes the diversity of communities an individual is able to access in her local network. If weighted,
    each individual label is included as many times as the weight. When using bi-directional neighbors, 
    weights across both edges are summed up, with a non-existent edge treated as having zero weight.
    Parameters
    ----------
    G_directed : NetworkX DiGraph
    edges2weights : dict of {tuple : int}
    weighted : bool
    Returns
    ------
    tuple of (dict, dict, int)
    """
    node2community = {node : i for i, c in enumerate(communities) for node in c}
    node2herf, node2herf_all_neighbors = defaultdict(lambda : None), defaultdict(lambda : None)
    if weighted:
        for u in G_directed:
            alter_communities = [node2community[v] for v in G_directed.neighbors(u) for _ in range(edges2weights[(u, v)])]
            node2herf[u] = sum([(counts/len(alter_communities))**2 for group, counts in Counter(alter_communities).items()])
            alter_communities = [node2community[v] for v in find_all_neighbors(G_directed, u) for _ in range(
                edges2weights.get((u, v), 0) + edges2weights.get((v, u), 0))]
            node2herf_all_neighbors[u] = sum([(counts/len(alter_communities))**2 for group, counts in Counter(alter_communities).items()])
            
    else:
        for u in G_directed:
            alter_communities = [node2community[v] for v in G_directed.neighbors(u)]
            node2herf[u] = sum([(counts/len(alter_communities))**2 for group, counts in Counter(alter_communities).items()])
            
            alter_communities = [node2community[v] for v in find_all_neighbors(G_directed, u)]
            node2herf_all_neighbors[u] = sum([(counts/len(alter_communities))**2 for group, counts in Counter(alter_communities).items()])
    return (node2herf, node2herf_all_neighbors)

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
    # pre-computing community membership so that the two methods share communities
    weighted_communities = algorithms.leiden(G_directed, weights='weight').communities
    unweighted_communities = algorithms.leiden(G_directed).communities

    node2ego_bridging_weighted, node2ego_bridging_all_neighbors_weighted = generate_ego_bridging(G_directed, edges2weights, weighted_communities, True)
    node2ego_bridging_unweighted, node2ego_bridging_all_neighbors_unweighted = generate_ego_bridging(G_directed, edges2weights, unweighted_communities, False)

    node2comm_bridging_weighted, node2comm_bridging_all_neighbors_weighted = generate_global_bridging(G_directed, edges2weights, weighted_communities, True)
    node2comm_bridging_unweighted, node2comm_bridging_all_neighbors_unweighted = generate_global_bridging(G_directed, edges2weights, unweighted_communities, False)

    node2comm_diversity_weighted, node2comm_diversity_all_neighbors_weighted = generate_community_diversity(G_directed, edges2weights, weighted_communities, True)
    node2comm_diversity_unweighted, node2comm_diversity_all_neighbors_unweighted = generate_community_diversity(G_directed, edges2weights, unweighted_communities, False)
    
    sys.stderr.write('Computing department-based bridging at %s.\n' % datetime.now())
    node2depart_bridging_weighted, node2depart_bridging_all_neighbors_weighted = generate_department_bridging(G_directed, quarter2usr2department[timekey], edges2weights, True)
    node2depart_bridging_unweighted, node2depart_bridging_all_neighbors_unweighted = generate_department_bridging(G_directed, quarter2usr2department[timekey], edges2weights, False)

    for n in G_directed:
        row = ([node2ego_bridging_unweighted[n], node2ego_bridging_all_neighbors_unweighted[n], node2comm_bridging_unweighted[n], node2comm_bridging_all_neighbors_unweighted[n], node2comm_diversity_unweighted[n], node2comm_diversity_all_neighbors_unweighted[n], len(unweighted_communities),
        node2ego_bridging_weighted[n], node2ego_bridging_all_neighbors_weighted[n], node2comm_bridging_weighted[n], node2comm_bridging_all_neighbors_weighted[n], node2comm_diversity_weighted[n], node2comm_diversity_all_neighbors_weighted[n], len(weighted_communities),
        node2depart_bridging_unweighted[n], node2depart_bridging_all_neighbors_unweighted[n], node2depart_bridging_weighted[n], node2depart_bridging_all_neighbors_weighted[n]])
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
        pool = multiprocessing.Pool(processes = num_cores, initializer = fill_demographics)
        results = [pool.apply_async(generate_network_measures, args=(timekey, edges, test_mode, )) for timekey, edges in time_edges.items()]
        pool.close()
        pool.join()
        network_measures = defaultdict(list)
        for r in results:
            network_measures.update(r.get())
    
    cols = (['ego_bridging_unweighted', 'ego_bridging_all_neighbors_unweighted', 'global_bridging_unweighted', 'global_bridging_all_neighbors_unweighted', 'comm_diversity_unweighted', 'comm_diversity_all_neighbors_unweighted', 'n_comm_unweighted',
    'ego_bridging_weighted', 'ego_bridging_all_neighbors_weighted', 'global_bridging_weighted', 'global_bridging_all_neighbors_weighted', 'comm_diversity_weighted', 'comm_diversity_all_neighbors_weighted', 'n_comm_weighted',
    'department_bridging_unweighted', 'department_bridging_all_neighbors_unweighted', 'department_bridging_weighted', 'department_bridging_all_neighbors_weighted'])
    
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

    fill_demographics()
    quarterly_edges = defaultdict(list)

    sys.stderr.write("Reading edges from edge file at %s.\n" % str(datetime.now()))
    with open(edges_file, 'r') as f:
        for line in f:
            tup = literal_eval(line)
            quarterly_edges[tup[0]].append((tup[1], tup[2]))

    sys.stderr.write("Calculating network measures at %s.\n" % str(datetime.now()))
    file_name = 'coco_diversity_bridging_all_test.csv' if test_mode else 'coco_diversity_bridging_all.csv'
    
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
