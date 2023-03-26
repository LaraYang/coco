#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    coco_network.py test|actual short|long unweighted|weighted|both
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
from networkx.algorithms.community import greedy_modularity_communities
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
email2user_id, user_id2race, user_id2gender, user_id2department = {}, {}, {}, {}

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
                if sender_userid in recipients_userids:
                    recipients_userids.remove(sender_userid)
                if timekey_type == 'quarter':
                    edges[to_quarter(activity['ActivityCreatedAt'], format='str')] += [(sender_userid, r) for r in recipients_userids]
                elif timekey_type == 'year':
                    edges[to_year(activity['ActivityCreatedAt'], format='str')] += [(sender_userid, r) for r in recipients_userids]
                else:
                    # all values have the same key under this condition; written this way to ensure consistency in type of object returned by current function
                    edges['all'] += [(sender_userid, r) for r in recipients_userids]
    return edges

def fill_demographics():
    survey_df = pd.read_csv(hr_file)[['UID', 'EEO Code', 'Gender', 'Department']]
    survey_df.set_index('UID', inplace=True)
    # no missing and additional revising needed
    uid2gender = survey_df['Gender'].to_dict()
    survey_df.dropna(subset=['EEO Code'], inplace=True)
    survey_df['EEO Code'] = survey_df['EEO Code'].apply(lambda s : 'Other' if ('Hispanic' in s or 'Black' in s or 'Hawaiian' in s) else s)
    survey_df['EEO Code'] = survey_df['EEO Code'].apply(lambda s : np.nan if 'missing' in s else s)
    uid2department = survey_df['Department'].dropna().to_dict()
    uid2race = survey_df['EEO Code'].dropna().to_dict()
    user_qualtrics_df = pd.read_csv(user_qualtrics_file)[['Email', 'UID']]
    user_qualtrics_df.set_index('Email', inplace=True)
    email2uid = user_qualtrics_df.to_dict()['UID']
    global user_id2race, user_id2gender, user_id2department
    for email, user_id in email2user_id.items():
        if email in email2uid:
            uid = email2uid[email]
            if uid in uid2gender: user_id2gender[user_id] = uid2gender[uid]
            if uid in uid2race: user_id2race[user_id] = uid2race[uid]
            if uid in uid2department: user_id2department[user_id] = uid2department[uid]
    return

def blau_index(population):
    """
    Calculate Blau Index, used for calculating diversity of based on qualitiative groups
    Equivalent to the Gini-Simpson Index
    Parameter
    ---------
    population : list
        A list of category labels
    Returns
    -------
    float
        Blau index of population
    """
    total = len(population)
    pop_counts = Counter(population)
    return (1 - sum([(counts/total)**2 for group, counts in pop_counts.items()]))

def generate_community_measures(G_directed):
    """
    Generate diversity and within group proportion for directed Graph
    Parameter
    ---------
    G_directed : NetworkX graph
    Returns
    -------
    tuple
        A 4-tuple of dictionaries matching node (i.e., user ids) to the average gender diversity, racial diversity of its communities, 
        as well as the average proportion of its ingroup members both in terms of gender and race
    """
    # supposedly defined for directed graphs but leads to errors in execution; thus using undirected graph
    # does not support edge weights
    communities = list(greedy_modularity_communities(G_directed.to_undirected()))
    community2genders = {community_name: [user_id2gender[n] for n in community_nodes if n in user_id2gender] for community_name, community_nodes in enumerate(communities)}
    community2gender_prop = {community_name : {gender : count/len(genders) for gender, count in Counter(genders).items()} for community_name, genders in community2genders.items()}
    community2races = {community_name: [user_id2race[n] for n in community_nodes if n in user_id2race] for community_name, community_nodes in enumerate(communities)}
    community2race_prop = {community_name : {race : count/len(races) for race, count in Counter(races).items()} for community_name, races in community2races.items()}

    community2diversity = {community_name : (blau_index(community2genders[community_name]),
        blau_index(community2races[community_name])) for community_name, community_nodes in enumerate(communities)}
    node2communities = defaultdict(list)
    for community_name, community_nodes in enumerate(communities):
        for n in community_nodes:
            node2communities[n].append(community_name)

    node2gender_diversity, node2gender_ingroup, node2race_diversity, node2race_ingroup = defaultdict(lambda : None), defaultdict(lambda : None), defaultdict(lambda : None), defaultdict(lambda : None)
    for n, communities in node2communities.items():
        total_race_diversity, total_gender_diversity, total_gender_ingroup, total_race_ingroup = 0, 0, 0, 0
        num_community = len(communities)
        for c in communities:
            gender_diversity, race_diversity = community2diversity[c]
            total_gender_diversity += gender_diversity
            total_race_diversity += race_diversity
            if n in user_id2gender:
                total_gender_ingroup += community2gender_prop[c][user_id2gender[n]]
            else:
                total_gender_ingroup = np.nan
            if n in user_id2race:
                total_race_ingroup += community2race_prop[c][user_id2race[n]]
            else:
                total_race_ingroup = np.nan
        node2gender_diversity[n], node2race_diversity[n], node2gender_ingroup[n], node2race_ingroup[n] = (total_gender_diversity/num_community, 
             total_race_diversity/num_community, total_gender_ingroup/num_community, total_race_ingroup/num_community)
    return (node2gender_diversity, node2gender_ingroup, node2race_diversity, node2race_ingroup)

def generate_network_composition(G_directed):
    """
    Generate diversity and within group proportion for directed Graph
    Parameter
    ---------
    G_directed : NetworkX graph
    Returns
    -------
    tuple
        A 4-tuple of dictionaries matching node (i.e., user ids) to the gender diversity, racial diversity of its ego-networks
        as well as proportion of ingroup members both in terms of gender and race
    """
    node2gender_diversity, node2gender_ingroup, node2race_diversity, node2race_ingroup, node2ei = defaultdict(lambda : None), defaultdict(lambda : None), defaultdict(lambda : None), defaultdict(lambda : None), defaultdict(lambda : None)
    for u in G_directed:
        # assumes if gender provided for user, race also is
        if u in user_id2gender or u in user_id2department:
            genders, races = [], []
            internal_links, external_links = 0, 0
            # As G is a directed network, u's network only includes those who u has sent an email to
            for v in G_directed.neighbors(u):
                if u in user_id2gender and v in user_id2gender:
                    genders.append(user_id2gender[v])
                if u in user_id2race and v in user_id2race:
                    races.append(user_id2race[v])
                if u in user_id2department and v in user_id2department:
                    if user_id2department[u] == user_id2department[v]:
                        internal_links += 1
                    else:
                        external_links += 1
            if u in user_id2gender and len(genders) > 0:
                node2gender_diversity[u] = blau_index(genders)
                node2gender_ingroup[u] = genders.count(user_id2gender[u]) / len(genders)
            if u in user_id2race and len(races) > 0:
                node2race_diversity[u] = blau_index(races)
                node2race_ingroup[u] = races.count(user_id2race[u]) / len(races)
            if u in user_id2department and (external_links > 0 or internal_links > 0):
                node2ei[u] = (external_links - internal_links) / (external_links + internal_links)
    return (node2gender_diversity, node2gender_ingroup, node2race_diversity, node2race_ingroup, node2ei)

def generate_network_measures(timekey, edges, weighted, test_mode, short_mode):
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
    short_mode : bool
        If true, constraint is not computed
    """
    if test_mode: edges = random.sample(edges, len(edges) // 500)

    G_directed = nx.DiGraph()
    if not weighted:
        sys.stderr.write('Generating unweighted network measures for %s with %d edges at %s.\n' % (timekey, len(edges), datetime.now()))
        edges = list(set(edges))
        G_directed.add_edges_from(edges)
    else:
        sys.stderr.write('Generating weighted network measures for %s with %d edges at %s.\n' % (timekey, len(edges), datetime.now()))
        edges2weights = Counter(edges)
        weighted_edges = [(edge[0], edge[1], weight) for edge, weight in edges2weights.items()]
        G_directed.add_weighted_edges_from(weighted_edges)
    network_measures = defaultdict(list)
    # direction and weight matters; when weighted, sum of incoming edge weights; additional weight argument has no effect when G_directed is unweighted
    indegree = G_directed.in_degree(weight='weight')
    # direction and weight matters; when weighted, sum of outgoing edge weights
    outdegree = G_directed.out_degree(weight='weight')
    # direction and weight matters; edges of weight = 0 are filtered out, which isn't possible in the current use-case
    betweenness_centrality = nx.betweenness_centrality(G_directed, weight='weight')
    # direction and weight matters; for a directed graph this is left eigenvector centrality which corresponds to the in-edges in the graph
    # thus, a highly eigen-central node is one with many incoming emails and many important people emailing them
    eigenvector_centrality = nx.eigenvector_centrality_numpy(G_directed, weight='weight')
    # direction and weight matters
    clustering = nx.clustering(G_directed, weight='weight')
    gender_diversity, gender_ingroup, race_diversity, race_ingroup, ei_index = generate_network_composition(G_directed)
    
    if not short_mode:
        # this differs from the handcoded version
        # pij denominator: normalized_mutual_weight (nx) normalizes by all edges while get_proportion normalize by directed edges only
        # pij numerator: mutual_weight counts edges in both directions while get_proportion only includes directed edges
        # Graph class in networkx does not allow for parallel edges while other network packages - networkit and snap - do
        # Converting multiple edges to weights allows us to take weight into account when computing constraint and clustering
        constraint = nx.constraint(G_directed, weight='weight')
        for n in G_directed:
            row = ([indegree[n], outdegree[n], betweenness_centrality[n], eigenvector_centrality[n], clustering[n], constraint[n], 
                gender_diversity[n], gender_ingroup[n], race_diversity[n], race_ingroup[n], ei_index[n]])
            if timekey == 'all':
                network_measures[n] = row
            else:
                network_measures[(n, timekey)] = row
    else:
        for n in G_directed:
            row = ([indegree[n], outdegree[n], betweenness_centrality[n], eigenvector_centrality[n], clustering[n],
                gender_diversity[n], gender_ingroup[n], race_diversity[n], race_ingroup[n], ei_index[n]])
            if timekey == 'all':
                network_measures[n] = row
            else:
                network_measures[(n, timekey)] = row
    return dict(network_measures)

def time_edges_to_df(time_edges, timekey_type, weighted=False, test_mode=False, short_mode=False):
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
    short_mode : bool, optional
        If true, do not compute constraint, a time-consuming measure
    Returns
    -------
    df : pd.DataFrame
        A dataframe of network measures with user id and timekey_type as index
    """
    if test_mode:
        test_timekey = random.choice(list(time_edges))
        sys.stderr.write("Testing timekey %s out of %d time periods.\n" % (test_timekey, len(time_edges)))
        network_measures = generate_network_measures(test_timekey, time_edges[test_timekey], weighted, test_mode, short_mode)
    else:
        pool = multiprocessing.Pool(processes = num_cores)
        results = [pool.apply_async(generate_network_measures, args=(timekey, edges, weighted, test_mode, short_mode, )) for timekey, edges in time_edges.items()]
        pool.close()
        pool.join()
        network_measures = defaultdict(list)
        for r in results:
            network_measures.update(r.get())
    if short_mode:
        cols = ['indegree', 'outdegree', 'betweenness_centrality', 'eigenvalue_centrality', 'clustering', 'gender_diversity', 'gender_ingroup', 'race_diversity', 'race_ingroup', 'ei_index']
    else:
        cols = ['indegree', 'outdegree', 'betweenness_centrality', 'eigenvalue_centrality', 'clustering', 'constraint', 'gender_diversity', 'gender_ingroup', 'race_diversity', 'race_ingroup', 'ei_index']
    if timekey_type == 'all':
        df = dict_to_df(network_measures, cols, index_name=['user_id'])    
    else:
        df = dict_to_df(network_measures, cols, index_name=['user_id', timekey_type])
    return df.round(5)

def extract_network_measures(timekey_type, file_name, test_mode=False, short_mode=False, weighted_mode='both'):
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
    short_mode : bool, optional
        Included to provide the option of reducing computing time by omitting time-intensive computations.
        If short_mode, constraint is not computed.
    weighted_mode : bool, optional
        Whether to build a weighted or unweighted network. If weighted, email frequency betweeen the two parties are used
        as measure of strength.
    """
    sys.stderr.write("--- Computing edges at %s ---\n" % str(datetime.now()))
    edges = get_edges(timekey_type)
    # needs to be called after compiling edges as this function fills email2uid dictionary
    fill_demographics()
    sys.stderr.write("--- Calculating network measures at %s ---\n" % str(datetime.now()))
    if test_mode and short_mode:
        file_name += "_test_short"
    elif test_mode:
        file_name += "_test"
    elif short_mode:
        file_name += "_short"

    if weighted_mode == 'weighted' or weighted_mode == 'both':
        weighted = True
        df = time_edges_to_df(edges, timekey_type, weighted, test_mode, short_mode)
        df_filename = os.path.join(output_dir, file_name+"_weighted.csv")
        df.to_csv(df_filename)
        sys.stderr.write("Finished outputting weighted network measures at %s.\n" % str(datetime.now()))

    if weighted_mode == 'unweighted' or weighted_mode == 'both':
        weighted = False
        df = time_edges_to_df(edges, timekey_type, weighted, test_mode, short_mode)
        df_filename = os.path.join(output_dir, file_name+"_unweighted.csv")
        df.to_csv(df_filename)
        sys.stderr.write("Finished outputting unweighted network measures at %s.\n" % str(datetime.now()))
    return

if __name__ == '__main__':
    starttime = datetime.now()
    test_mode, short_mode, weighted_mode = False, False, 'both'
    try:
        test_mode = sys.argv[1].lower() == 'test'
        short_mode = sys.argv[2].lower() == 'short'
        weighted_mode = sys.argv[3].lower()
    except IndexError as error:
        pass
    sys.stderr.write('Generating Network Measures at %s.\n' % datetime.now())
#    extract_network_measures('quarter', 'network_quarterly_internal', test_mode, short_mode, weighted_mode)
    extract_network_measures('quarter', 'network_quarterly_internal', test_mode, short_mode, weighted_mode)
    sys.stderr.write("Finished running at %s, with a duration of %s.\n"
        % (str(datetime.now()), str(datetime.now() - starttime)))
