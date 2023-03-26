#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    coco_network_embed.py test|actual
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
def get_quarterly_edges(test_mode):
    """
    Uses activity_file and users_file to return a list of edges at the quarter level.
    Nodes are named by User Ids and not email addresses, which is why users_file is necessary.
    This file does not constrain measures to target users only as all communications
    to employees with or without survey data should be included in computation
    Returns
    -------
    edges : dict
        A dictionary mapping quarters to lists of 2-tuples that represent directed edges
    """
    global email2user_id
    with open(users_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if test_mode and i > lines_to_test: break
            user = json.loads(line)
            for e in user['Emails']:
                email2user_id[e] = user['UserId']

    edges = defaultdict(list)
    with open(activity_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if test_mode and i > lines_to_test: break
            activity = json.loads(line)
            sender_userid = activity["UserId"]
            recipients = get_recipients(activity)
            pure_internal, pure_external = True, True
            if test_mode and i > lines_to_test:
                break
            for r in recipients:
                domain = r.split('@')[1]
                if pure_external and domain in collabera_hashes:
                    pure_external = False
                elif pure_internal and domain not in collabera_hashes:
                    pure_internal = False
            if pure_internal:
                recipients_userids = list(set([email2user_id[r] for r in recipients if r in email2user_id]))
                if sender_userid in recipients_userids:
                    recipients_userids.remove(sender_userid)
                quarter = to_quarter(activity['ActivityCreatedAt'], format='str')
                edges[quarter] += [(sender_userid, r) for r in recipients_userids]

    return edges

def generate_network_embeddedness(G_directed, weighted, edges2weights):
    """
    Generate embeddedness of one's network
    Parameter
    ---------
    G_directed : NetworkX graph
    weighted : bool
        Whether to engage in weighted computations
    edges2weights : dict of {tuple : int}
        Maps directed edges to weights
    Returns
    -------
        Three different measures of network embeddedness at ego, alter, and alter's alter levels.
    """
    node2embed_ego, node2embed_alter, node2embed_alter2 = defaultdict(lambda : None), defaultdict(lambda : None), defaultdict(lambda : None)
    for u in G_directed:
        local_network = set(G_directed.neighbors(u))
        within_cluster, without_cluster = 0, 0
        # As G is a directed network, u's network only includes those who u has sent an email to
        for v in G_directed.neighbors(u):
            for w in G_directed.neighbors(v):
                add = edges2weights[v, w] if weighted else 1
                if w in local_network:
                    within_cluster += add
                else:
                    without_cluster += add

        if (without_cluster+within_cluster) == 0:
            node2embed_ego[u] = np.nan
        else:
            # if 1, no within_cluster ties, if -1, all within-cluster ties
            node2embed_ego[u] = (without_cluster-within_cluster)/(without_cluster+within_cluster)

    # alter level measure
    for u in G_directed:
        local_network = set(G_directed.neighbors(u))
        v2ei, uv2weight = {}, {}
        # As G is a directed network, u's network only includes those who u has sent an email to
        for v in G_directed.neighbors(u):
            uv2weight[v] = edges2weights[u, v]
            # skip any alter who doesn't have an alter
            if len(list(G_directed.neighbors(v))) == 0:
                continue
            within_cluster, without_cluster = 0, 0    
            for w in G_directed.neighbors(v):
                add = edges2weights[v, w] if weighted else 1
                if w in local_network:
                    within_cluster += add
                else:
                    without_cluster += add
            v2ei[v] = (without_cluster-within_cluster)/(without_cluster+within_cluster)
        
        if len(v2ei) == 0:
            node2embed_alter[u] = np.nan
        else:
            u_ei = 0
            if weighted:
                u_weight = sum(uv2weight.values())
                uv2prop = {v : weight / u_weight for v, weight in uv2weight.items()}
                for v, ei in v2ei.items():
                    u_ei += (uv2prop[v] * ei)
            else:
                u_ei = sum(v2ei.values()) / len(v2ei)
            node2embed_alter[u] = u_ei

        
    # alter alter level measure
    for u in G_directed:
        local_network = set(G_directed.neighbors(u))
        v2ei, uv2weight = {}, {}
        # As G is a directed network, u's network only includes those who u has sent an email to
        for v in G_directed.neighbors(u):
            uv2weight[v] = edges2weights[u, v]
            # skip any alter who doesn't have an alter
            if len(list(G_directed.neighbors(v))) == 0:
                continue
            w2ei, vw2weight = {}, {}
            for w in G_directed.neighbors(v):
                vw2weight[w] = edges2weights[v, w]
                if len(list(G_directed.neighbors(w))) == 0:
                    continue
                within_cluster, without_cluster = 0, 0
                for z in G_directed.neighbors(w):
                    add = edges2weights[w, z] if weighted else 1
                    if z in local_network:
                        within_cluster += add
                    else:
                        without_cluster += add
                w2ei[w] = (without_cluster-within_cluster)/(without_cluster+within_cluster)
            
            if len(w2ei) > 0:
                v_ei = 0
                if weighted:
                    v_weight = sum(vw2weight.values())
                    vw2prop = {w : weight/v_weight for w, weight in vw2weight.items()}
                    for w, ei in w2ei.items():
                        v_ei += (vw2prop[w] * ei)
                else:
                    v_ei = sum(w2ei.values()) / len(w2ei)
                v2ei[v] = v_ei
        
        if len(v2ei) == 0:
            node2embed_alter2[u] = np.nan
        else:
            u_ei = 0
            if weighted:
                u_weight = sum(uv2weight.values())
                uv2prop = {u : weight / u_weight for u, weight in uv2weight.items()}
                for v, ei in v2ei.items():
                    u_ei += (uv2prop[v] * ei)
            else:
                u_ei = sum(v2ei.values()) / len(v2ei)
            node2embed_alter2[u] = u_ei

    return [node2embed_ego, node2embed_alter, node2embed_alter2]

def generate_community_ei(G_directed, node2community, weighted, edges2weights):
    """
    Calculate EI index based on community structure.
    Parameter
    ---------
    G_directed : NetworkX graph
    node2community : dict of {str : int}
        A dictionary mapping user IDs to integers that represent distinct communities
    weighted : bool
        Whether to engage in weighted computations
    edges2weights : dict of {tuple : int}
        Maps directed edges to weights
    Returns
    -------
        Six different measures of network embeddedness at ego, alter, and alter's alter levels, with optional
        filtering.
    """
    node2embed_ego, node2embed_ego_filter, node2embed_alter, node2embed_alter_filter, node2embed_alter2, node2embed_alter2_filter = [defaultdict(lambda : None) for _ in range(6)]
    for u in G_directed:
        local_community = node2community[u]
        local_network = set(G_directed.neighbors(u))
        within_cluster, within_cluster_filter, without_cluster, without_cluster_filter = 0, 0, 0, 0
        for v in G_directed.neighbors(u):
            filtered = node2community[v] == local_community
            for w in G_directed.neighbors(v):
                add = edges2weights[v, w] if weighted else 1
                if w in local_network or node2community[w] == local_community:
                    within_cluster += add
                    if filtered:
                        within_cluster_filter += add
                else:
                    without_cluster += add
                    if filtered:
                        without_cluster_filter += add

        if (without_cluster+within_cluster) == 0:
            node2embed_ego[u] = np.nan
        else:
            node2embed_ego[u] = (without_cluster-within_cluster)/(without_cluster+within_cluster)
        
        if (without_cluster_filter + within_cluster_filter) == 0:
            node2embed_ego_filter[u] = np.nan
        else:
            node2embed_ego_filter[u] = (without_cluster_filter-within_cluster_filter)/(without_cluster_filter+within_cluster_filter) 

    for u in G_directed:
        local_community = node2community[u]
        local_network = set(G_directed.neighbors(u))
        v2ei, v2ei_filtered, uv2weight = {}, {}, {}
        # As G is a directed network, u's network only includes those who u has sent an email to
        for v in G_directed.neighbors(u):
            uv2weight[v] = edges2weights[u, v]
            filtered = node2community[v] == local_community
            # skip any alter who doesn't have an alter
            if len(list(G_directed.neighbors(v))) == 0:
                continue
            within_cluster, without_cluster = 0, 0    
            for w in G_directed.neighbors(v):
                add = edges2weights[v, w] if weighted else 1
                if w in local_network or node2community[w] == local_community:
                    within_cluster += add
                else:
                    without_cluster += add
            v_ei = (without_cluster-within_cluster)/(without_cluster+within_cluster)
            v2ei[v] = v_ei
            if node2community[v] == local_community:
                v2ei_filtered[v] = v_ei
        
        if len(v2ei) == 0:
            node2embed_alter[u] = np.nan
        else:
            u_ei = 0
            if weighted:
                u_weight = sum(uv2weight.values())
                uv2prop = {u : weight / u_weight for u, weight in uv2weight.items()}
                for v, ei in v2ei.items():
                    u_ei += (uv2prop[v] * ei)
            else:
                u_ei = sum(v2ei.values()) / len(v2ei)
            node2embed_alter[u] = u_ei
        
        if len(v2ei_filtered) == 0:
            node2embed_alter_filter[u] = np.nan
        else:
            u_ei_filtered = 0 
            # if weight by total traffic to same community alters only, need to create a new uv2prop
            if weighted: 
                for v, ei in v2ei_filtered.items():
                    u_ei_filtered += (uv2prop[v] * ei)
            else:
                u_ei_filtered = sum(v2ei_filtered.values()) / len(v2ei_filtered)
            node2embed_alter_filter[u] = u_ei_filtered

    # alter alter level measure
    for u in G_directed:
        local_network = set(G_directed.neighbors(u))
        local_community = node2community[u]
        v2ei, v2ei_filtered, uv2weight = {}, {}, {}
        # As G is a directed network, u's network only includes those who u has sent an email to
        for v in G_directed.neighbors(u):
            uv2weight[v] = edges2weights[u, v]
            # skip any alter who doesn't have an alter
            if len(list(G_directed.neighbors(v))) == 0:
                continue
            w2ei, vw2weight = {}, {}
            for w in G_directed.neighbors(v):
                vw2weight[w] = edges2weights[v, w]
                if len(list(G_directed.neighbors(w))) == 0:
                    continue
                within_cluster, without_cluster = 0, 0
                for z in G_directed.neighbors(w):
                    add = edges2weights[w, z] if weighted else 1
                    if z in local_network or node2community[z] == local_community:
                        within_cluster += add
                    else:
                        without_cluster += add
                w2ei[w] = (without_cluster-within_cluster)/(without_cluster+within_cluster)
            
            if len(w2ei) > 0:
                v_ei = 0
                if weighted:
                    v_weight = sum(vw2weight.values())
                    vw2prop = {w : weight/v_weight for w, weight in vw2weight.items()}
                    for w, ei in w2ei.items():
                        v_ei += (vw2prop[w] * ei)
                else:
                    v_ei = sum(w2ei.values()) / len(w2ei)
                v2ei[v] = v_ei
                if node2community[v] == local_community:
                    v2ei_filtered[v] = v_ei
        if len(v2ei) == 0:
            node2embed_alter2[u] = np.nan
        else:
            u_ei = 0
            if weighted:
                u_weight = sum(uv2weight.values())
                uv2prop = {u : weight / u_weight for u, weight in uv2weight.items()}
                for v, ei in v2ei.items():
                    u_ei += (uv2prop[v] * ei)
            else:
                u_ei = sum(v2ei.values()) / len(v2ei)
            node2embed_alter2[u] = u_ei

        if len(v2ei_filtered) == 0:
            node2embed_alter2[u] = np.nan
        else:
            u_ei_filtered = 0
            if weighted:
                for v, ei in v2ei_filtered.items():
                    u_ei += (uv2prop[v] * ei)
            else:
                u_ei = sum(v2ei_filtered.values()) / len(v2ei)
            node2embed_alter2_filter[u] = u_ei

    return [node2embed_ego, node2embed_ego_filter, node2embed_alter, node2embed_alter_filter, node2embed_alter2, node2embed_alter2_filter]
    
def generate_community_embeddedness(G_directed, community_algorithm, weighted, edges2weights):
    """
    Generate the degree to which one's local network is embedded in one's community or outside of one's community
    Parameter
    ---------
    G_directed : NetworkX DiGraph
    community_algorithm : str
        Indicates type of algorithm to use for community detection
    weight : str
        Either None or weight attribute
    Returns
    -------
    measures : [dict, dict, dict, dict, dict, dict, dict, n_comm]
        A list of possible community measures
    """
    sys.stderr.write("Computing communities using {}'s algorithm at {}.\n".format(community_algorithm, datetime.now()))
    communities = []
    if community_algorithm == 'cnm':
        communities = algorithms.greedy_modularity(G_directed, weight='weight').communities
    elif community_algorithm == 'leiden':
        try:
            communities = algorithms.leiden(G_directed, weights='weight').communities
        except nx.exception.AmbiguousSolution as e:
            print('No community found using {} due to AmbiguousSolution error.'.format(community_algorithm))
            return defaultdict(lambda : np.nan)
    elif community_algorithm == 'surprise':
        try:
            communities = algorithms.surprise_communities(G_directed, weights='weight').communities
        except nx.exception.AmbiguousSolution as e:
            print('No community found using {} due to AmbiguousSolution error.'.format(community_algorithm))
            return defaultdict(lambda : np.nan)
    else:
        print("Community detection algorithm {} not supported".format(community_algorithm))
        return defaultdict(lambda : np.nan)
    
    node2community = {node : i for i, c in enumerate(communities) for node in c}
    measures = generate_community_ei(G_directed, node2community, weighted, edges2weights)
    measures.append(len(communities))
    return measures

def compute_threshold(edges2weights):
    """
    Computes the 20th percentage edge weight for all nodes
    Parameters
    ----------
    edges2weights : dict of {tuple : list}
        Maps all directed edges to the weight of the edge
    Returns
    -------
    node2threshold : dict of {str : int}
        Maps all nodes to the 20th percentile threshold
    """
    node2weights = defaultdict(list)
    node2threshold = defaultdict(lambda : None)
    for edge, weight in edges2weights.items():
        node2weights[edge[0]].append(weight)

    for n, weights in node2weights.items():
        node2threshold[n] = np.percentile(weights, 20)
    return node2threshold

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

    weighted_degree = G_directed.degree(weight='weight')
    unweighted_degree = G_directed.degree(weight=None)
    weighted_clustering = nx.clustering(G_directed, weight='weight')
    unweighted_clustering = nx.clustering(G_directed, weight=None)
    
    sys.stderr.write('Computing network-based connectivity at %s.\n' % datetime.now())
    unweighted_node2embed_ego, unweighted_node2embed_alter, unweighted_node2embed_alter2 = generate_network_embeddedness(G_directed, False, edges2weights)
    weighted_node2embed_ego, weighted_node2embed_alter, weighted_node2embed_alter2 = generate_network_embeddedness(G_directed, True, edges2weights)
    sys.stderr.write('Computing community-based connectivity at %s.\n' % datetime.now())
    unweighted_comm_measures = generate_community_embeddedness(G_directed, 'leiden', False, edges2weights)
    unweighted_node2comm_ego, unweighted_node2comm_ego_filter, unweighted_node2comm_alter, unweighted_node2comm_alter_filter, unweighted_node2comm_alter2, unweighted_node2comm_alter2_filter, unweighted_n_comm = unweighted_comm_measures
    weighted_comm_measures = generate_community_embeddedness(G_directed, 'leiden', True, edges2weights)
    weighted_node2comm_ego, weighted_node2comm_ego_filter, weighted_node2comm_alter, weighted_node2comm_alter_filter, weighted_node2comm_alter2, weighted_node2comm_alter2_filter, weighted_n_comm = weighted_comm_measures
    
    if build_threshold_network:
        node2threshold = compute_threshold(edges2weights)
        G_directed_thres = nx.DiGraph()
        sys.stderr.write('Generating weighted network measures with threshold for %s with %d edges at %s.\n' % (timekey, len(edges), datetime.now()))
        weighted_edges_thres = [(edge[0], edge[1], weight) for edge, weight in edges2weights.items() if weight > node2threshold[edge[0]]]
        G_directed_thres.add_weighted_edges_from(weighted_edges_thres)
        # if using a thresholded network, define variables here

    for n in G_directed:
        row = ([weighted_degree[n], unweighted_degree[n], weighted_clustering[n], unweighted_clustering[n],
            unweighted_node2embed_ego[n], unweighted_node2embed_alter[n], unweighted_node2embed_alter2[n],
            weighted_node2embed_ego[n], weighted_node2embed_alter[n], weighted_node2embed_alter2[n], 
            unweighted_node2comm_ego[n], unweighted_node2comm_ego_filter[n], unweighted_node2comm_alter[n], unweighted_node2comm_alter_filter[n], unweighted_node2comm_alter2[n], unweighted_node2comm_alter2_filter[n], unweighted_n_comm,
            weighted_node2comm_ego[n], weighted_node2comm_ego_filter[n], weighted_node2comm_alter[n], weighted_node2comm_alter_filter[n], weighted_node2comm_alter2[n], weighted_node2comm_alter2_filter[n], weighted_n_comm])
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
    
    cols = (['weighted_degree', 'unweighted_degree', 'weighted_clustering', 'unweighted_clustering',
        'unweighted_network_connectivity_ego', 'unweighted_network_connectivity_alter', 'unweighted_network_connectivity_alter2',
        'weighted_network_connectivity_ego', 'weighted_network_connectivity_alter', 'weighted_network_connectivity_alter2',
        'unweighted_community_connectivity_ego', 'unweighted_community_connectivity_ego_filter', 'unweighted_community_connectivity_alter', 'unweighted_community_connectivity_alter_filter',
        'unweighted_community_connectivity_alter2', 'unweighted_community_connectivity_alter2_filter', 'unweighted_n_comm',
        'weighted_community_connectivity_ego', 'weighted_community_connectivity_ego_filter', 'weighted_community_connectivity_alter', 'weighted_community_connectivity_alter_filter',
        'weighted_community_connectivity_alter2', 'weighted_community_connectivity_alter2_filter', 'weighted_n_comm'])

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
    if os.path.exists(edges_file):
        sys.stderr.write("Reading edges from edge file at %s.\n" % str(datetime.now()))
        with open(edges_file, 'r') as f:
            for line in f:
                tup = literal_eval(line)
                quarterly_edges[tup[0]].append((tup[1], tup[2]))
    else:
        sys.stderr.write("Computing edges at %s.\n" % str(datetime.now()))
        quarterly_edges = get_quarterly_edges(test_mode)
        sys.stderr.write("Writing edges to edge file at %s.\n" % str(datetime.now()))
        with open(edges_file, 'w') as f:
            for quarter, edges in quarterly_edges.items():
                for e in edges:
                    f.write(str((quarter, e[0], e[1]))+'\n')

    sys.stderr.write("Calculating network measures at %s.\n" % str(datetime.now()))
    file_name = 'coco_network_embedded_test.csv' if test_mode else 'coco_network_embedded.csv'
    
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
