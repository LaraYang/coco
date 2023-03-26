#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    coco_network.py test|actual weighted|unweighted|both
"""
from collections import defaultdict, Counter
from datetime import datetime
import sys
import numpy as np
import pandas as pd
import snap
import multiprocessing
import os
import random
import networkx as nx
import ujson as json
from utils import *

# 12 cores took about 3-4 days, 16 cores took more than 2 days just on weighted
num_cores = 16
home_dir = "/ifs/projects/amirgo-identification/"
email_dir = os.path.join(home_dir, "email_data/")
email_file = os.path.join(email_dir, 'MessagesHashed.jsonl')
users_file = os.path.join(email_dir, 'Users.json')
activity_file = os.path.join(email_dir, 'Activities.json')
output_dir = os.path.join(home_dir, "email_idtf_data")

survey_filename = "/ifs/gsb/amirgo/spacespace/spacespace/Coco/analyses_data/preprocessed_survey_hr.csv"
user_qualtrics_file = os.path.join(home_dir, "survey_hr_data", "UsersQualtrics.csv")
hr_file = os.path.join(home_dir, "survey_hr_data", "Collabera_HR_Perf.csv")
email2user_id, sid2sentiment = {}, {}

domain_hash = {
    'collabera.com':                     '509c8f6b1127bceefd418c023533d653', 
    'collaberainc.mail.onmicrosoft.com': 'ec5b67548b6ec06f1234d198efec741e', 
    'collaberainc.onmicrosoft.com':      '86160680578ee9258f097a67a5f25af9', 
    'collaberasp.com':                   '6bf3934d19f1acf5b9295b63e0e7f66e', 
    'g-c-i.com':                         '3444d1f7d5e46443080f2d069e41a10c'}
collabera_hashes = set([v for k, v in domain_hash.items()])

def get_edges(timekey_type):
    """
    Uses activity_file and users_file to return a list of edges at the quarter level.
    Nodes are named by User Ids and not email addresses, which is why users_file is necessary.
    This file does not constrain measures to target users only as all communications
    to employees with or without survey data should be included in computation
    Parameters
    ----------
    timekey_type : str
        Specifies how to aggregate edges: per year, per quarter, or all edges. Note the switch in 
        language - time_key represents actual time periods, timekey_type represents the type of time key
        - quarter, year, or no time key needed
    Returns
    -------
    edges : dict
        A dictionary mapping timekeys to lists of 2-tuples that represent directed edges
    """
    global email2user_id
    with open(users_file, encoding='utf-8') as f:
        for line in f:
            user = json.loads(line)
            for e in user['Emails']:
                email2user_id[e] = user['UserId']
    edges = defaultdict(list)
    with open(email_file, encoding='utf-8') as f:
        for line in f:
            message = json.loads(line)
            liwc = message['liwc']
            posemo = liwc['Posemo'] if 'Posemo' in liwc else 0
            negemo = liwc['Negemo'] if 'Negemo' in liwc else 0
            num_tok = len(message['hb'].replace('\n', ' ').replace('SENT_END', '').split())
            sid2sentiment[message['sid']] = (posemo, negemo, num_tok)
    with open(activity_file, encoding='utf-8') as f:
        for line in f:
            activity = json.loads(line)
            sender_userid = activity["UserId"]
            recipients = get_recipients(activity)
            pure_internal, pure_external = True, True
            for r in recipients:
                domain = r.split('@')[1]
                if pure_external and domain in collabera_hashes:
                    pure_external = False
                elif pure_internal and domain not in collabera_hashes:
                    pure_internal = False
            if pure_internal:
                recipients_userids = list(set([email2user_id[r] for r in recipients if r in email2user_id]))
                pos_emo, neg_emo, num_tok = sid2sentiment[activity['MailSummarySid']]
                if sender_userid in recipients_userids:
                    recipients_userids.remove(sender_userid)
                if timekey_type == 'quarter':
                    edges[to_quarter(activity['ActivityCreatedAt'], format='str')] += [(sender_userid, r, pos_emo, neg_emo, num_tok) for r in recipients_userids]
                elif timekey_type == 'year':
                    edges[to_year(activity['ActivityCreatedAt'], format='str')] += [(sender_userid, r, pos_emo, neg_emo, num_tok) for r in recipients_userids]
                else:
                    # all values have the same key under this condition; written this way to ensure consistency in type of object returned by current function
                    edges['all'] += [(sender_userid, r, pos_emo, neg_emo) for r in recipients_userids]
    return edges

def generate_network_measures(timekey, edges, weighted, test_mode):
    """
    Generating network measures for a given time period using edges
    Parameters
    ----------
    timekey : str
        A string that represents the time period for which network measures are being computed
    edges : list
        A list of directd edges represented by 2-tuples
    weighted : bool
        If not true, restrict edges to unique edges
    test_mode : bool
        If true, restrict edges to a hundredth of all edges
    """
    if test_mode: edges = random.sample(edges, len(edges) // 500)

    G_pos, G_neg, G_pos_norm, G_neg_norm = nx.DiGraph(), nx.DiGraph(), nx.DiGraph(), nx.DiGraph()
    
    if not weighted:
        sys.stderr.write('Unweighted network is not implemented')
    else:
        sys.stderr.write('Generating weighted network measures for %s with %d edges at %s.\n' % (timekey, len(edges), datetime.now()))
        edges2pos_weight = defaultdict(int)
        edges2neg_weight = defaultdict(int)
        edges2num_tok = defaultdict(int)
        for l in edges:
            num_tok = l[4]
            # Ignores liwc counts in subjects as they are often repeats of subjects of original emails
            if num_tok > 0:
                edge = (l[0], l[1])
                edges2pos_weight[edge] += l[2]
                edges2neg_weight[edge] += l[3]
                edges2num_tok[edge] += num_tok
        pos_weighted_edges = [(edge[0], edge[1], weight) for edge, weight in edges2pos_weight.items()]
        pos_weighted_norm_edges = [(edge[0], edge[1], weight/edges2num_tok[edge]) for edge, weight in edges2pos_weight.items()]
        neg_weighted_edges = [(edge[0], edge[1], weight) for edge, weight in edges2neg_weight.items()]
        neg_weighted_norm_edges = [(edge[0], edge[1], weight/edges2num_tok[edge]) for edge, weight in edges2neg_weight.items()]
        G_pos.add_weighted_edges_from(pos_weighted_edges)
        G_neg.add_weighted_edges_from(neg_weighted_edges)
        G_pos_norm.add_weighted_edges_from(pos_weighted_norm_edges)
        G_neg_norm.add_weighted_edges_from(neg_weighted_norm_edges)
    
    network_measures = defaultdict(list)
    pos_betweenness_centrality = defaultdict(lambda : np.nan, nx.betweenness_centrality(G_pos, weight='weight'))
    pos_eigenvector_centrality = defaultdict(lambda : np.nan, nx.eigenvector_centrality_numpy(G_pos, weight='weight'))
    pos_clustering = defaultdict(lambda : np.nan, nx.clustering(G_pos, weight='weight'))
    
    pos_norm_betweenness_centrality = defaultdict(lambda : np.nan, nx.betweenness_centrality(G_pos_norm, weight='weight'))
    pos_norm_eigenvector_centrality = defaultdict(lambda : np.nan, nx.eigenvector_centrality_numpy(G_pos_norm, weight='weight'))
    pos_norm_clustering = defaultdict(lambda : np.nan, nx.clustering(G_pos_norm, weight='weight'))
    
    neg_betweenness_centrality = defaultdict(lambda : np.nan, nx.betweenness_centrality(G_neg, weight='weight'))
    neg_eigenvector_centrality = defaultdict(lambda : np.nan, nx.eigenvector_centrality_numpy(G_neg, weight='weight'))
    neg_clustering = defaultdict(lambda : np.nan, nx.clustering(G_neg, weight='weight'))
    
    neg_norm_betweenness_centrality = defaultdict(lambda : np.nan, nx.betweenness_centrality(G_neg_norm, weight='weight'))
    neg_norm_eigenvector_centrality = defaultdict(lambda : np.nan, nx.eigenvector_centrality_numpy(G_neg_norm, weight='weight'))
    neg_norm_clustering = defaultdict(lambda : np.nan, nx.clustering(G_neg_norm, weight='weight'))

    all_nodes = set(list(G_pos.nodes())+list(G_neg.nodes()))
    for n in all_nodes:
        row = ([pos_betweenness_centrality[n], pos_eigenvector_centrality[n], pos_clustering[n],
            pos_norm_betweenness_centrality[n], pos_norm_eigenvector_centrality[n], pos_norm_clustering[n],
            neg_betweenness_centrality[n], neg_eigenvector_centrality[n], neg_clustering[n],
            neg_norm_betweenness_centrality[n], neg_norm_eigenvector_centrality[n], neg_norm_clustering[n]])
        if timekey == 'all':
            network_measures[n] = row
        else:
            network_measures[(n, timekey)] = row
    return dict(network_measures)

def time_edges_to_df(time_edges, timekey_type, weighted=False, test_mode=False):
    """
    Calculates network measures using edge lists
    Parameters
    ----------
    time_edges : dict
        A dictionary that maps timekeys to a list of edges belonging to that time period
    timekey_type : str
        Type of timekey - quarter, year, or all. If quarter or year, used as column name to specify name of time index in dataframe
    weighted : bool, optional
        If true, parallel edges are allowed
    test_mode : bool, optional
        If true, only generate one network
    Returns
    -------
    df : pd.DataFrame
        A dataframe of network measures with user id and timekey_type as index
    """
    if test_mode:
        test_timekey = random.choice(list(time_edges))
        sys.stderr.write("Testing timekey %s out of %d time periods.\n" % (test_timekey, len(time_edges)))
        network_measures = generate_network_measures(test_timekey, time_edges[test_timekey], weighted, test_mode)
    else:
        pool = multiprocessing.Pool(processes = num_cores)
        results = [pool.apply_async(generate_network_measures, args=(timekey, edges, weighted, test_mode, )) for timekey, edges in time_edges.items()]
        pool.close()
        pool.join()
        network_measures = defaultdict(list)
        for r in results:
            network_measures.update(r.get())
    
    cols = (['pos_betweenness_centrality', 'pos_eigenvalue_centrality', 'pos_clustering', 'pos_norm_betweenness_centrality', 'pos_norm_eigenvalue_centrality', 'pos_norm_clustering',
        'neg_betweenness_centrality', 'neg_eigenvalue_centrality', 'neg_clustering', 'neg_norm_betweenness_centrality', 'neg_norm_eigenvalue_centrality', 'neg_norm_clustering'])
    if timekey_type == 'all':
        df = dict_to_df(network_measures, cols, index_name=['user_id'])    
    else:
        df = dict_to_df(network_measures, cols, index_name=['user_id', timekey_type])
    return df.round(5)

def extract_network_measures(timekey_type, file_name, test_mode=False, weighted_mode='both'):
    """
    Main workhorse function for computing netwrork measures and writing them to file.
    Parameters
    ----------
    timekey_type : str
        Type of time period to be examined, one of 'quarter', 'year', or 'all'.
    file_name : str
        Prefix of output file name, of the format 'netowkr_{timekey}_{internal|external}'
    test_mode : bool, optional
        If testing, modify file_name to include flags for testing in final output file name
    weighted_mode : bool, optional
        Whether to build a weighted or unweighted network. If weighted, email frequency betweeen the two parties are used
        as measure of strength.
    """
    sys.stderr.write("--- Computing edges at %s ---\n" % str(datetime.now()))
    edges = get_edges(timekey_type)
    # needs to be called after compiling edges as this function fills email2uid dictionary
    sys.stderr.write("--- Calculating network measures at %s ---\n" % str(datetime.now()))
    if test_mode:
        file_name += "_test"
    
    if weighted_mode == 'weighted' or weighted_mode == 'both':
        weighted = True
        df = time_edges_to_df(edges, timekey_type, weighted, test_mode)
        df_filename = os.path.join(output_dir, file_name+"_weighted.csv")
        df.to_csv(df_filename)
        sys.stderr.write("Finished outputting weighted network measures at %s.\n" % str(datetime.now()))

    if weighted_mode == 'unweighted' or weighted_mode == 'both':
        weighted = False
        df = time_edges_to_df(edges, timekey_type, weighted, test_mode)
        df_filename = os.path.join(output_dir, file_name+"_unweighted.csv")
        df.to_csv(df_filename)
        sys.stderr.write("Finished outputting unweighted network measures at %s.\n" % str(datetime.now()))
    return

if __name__ == '__main__':
    starttime = datetime.now()
    test_mode, weighted_mode = False, 'both'
    try:
        test_mode = sys.argv[1].lower() == 'test'
        weighted_mode = sys.argv[2].lower()
    except IndexError as error:
        pass
    sys.stderr.write('Generating Network Measures at %s.\n' % datetime.now())
    extract_network_measures('quarter', 'network_quarterly_sentiment', test_mode, weighted_mode)
    sys.stderr.write("Finished running at %s, with a duration of %s.\n"
        % (str(datetime.now()), str(datetime.now() - starttime)))
