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

num_cores = 16
user_filter = True
compute_pair_distances = True
compute_centroid = False

home_dir = "/zfs/projects/faculty/amirgo-identification/"
email_dir = os.path.join(home_dir, "email_data/")
email_file = os.path.join(email_dir, 'MessagesHashed.jsonl')
users_file = os.path.join(email_dir, 'Users.json')
activity_file = os.path.join(email_dir, 'Activities.json')
output_dir = os.path.join(home_dir, "coco_email_idtf_data/")
mittens_dir = os.path.join(home_dir, "mittens")
centroid_file = os.path.join(output_dir, 'coco_embedding_centroids.csv')

user_qualtrics_file = os.path.join(home_dir, "survey_hr_data", "UsersQualtrics.csv")
long_hr_file = "/zfs/projects/faculty/amirgo-transfer/spacespace/spacespace/Coco/analyses_data/longitudinal_hr.csv"
embeddings_dir = os.path.join(home_dir, "mittens", "embeddings_high_prob_eng_08_50d_mincount300")

email2user_id, userid2race, userid_quarter2age, userid_quarter2cohort_year, userid_quarter2department, userid_quarter2title, userid_quarter2location, userid_quarter2country = [{} for i in range(8)]
usr_quarter2liwc, usr_quarter2dist, usr_quarter2centroid = defaultdict(lambda : Counter()), defaultdict(dict), defaultdict(list)

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
    global email2user_id, usr_quarter2liwcs, usr_quarter2dist
    sid2liwc = defaultdict(dict)
    
    with open(users_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if test_mode and i > lines_to_test: break
            user = json.loads(line)
            for e in user['Emails']:
                email2user_id[e] = user['UserId']
    
    with open(email_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if test_mode and i > lines_to_test: break
            email = json.loads(line)
            sid2liwc[email['sid']] = email['liwc']

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
                usr_quarter2liwc[sender_userid, quarter].update(Counter(sid2liwc[activity["MailSummarySid"]]))
    
    for key, liwc in usr_quarter2liwc.items():
        usr_quarter2dist[key] = counts2dist(liwc)
    return edges

def process_single_embedding_file(i, num_files, embeddings_dir, file):
    """
    Reading from one embedding file
    Parameters
    ----------
    i : int
        Index used for progress tracking
    num_files : int
        Total number of files to process used for progress tracking
    embeddings_dir : str
        Directory in which embedding files reside
    file : str
        Embedding file to open and process
    Returns
    -------
    float
        Centroid of all emebddings
    """
    mittens_file = os.path.join(embeddings_dir, file)
    try:
        model = KeyedVectors.load_word2vec_format(mittens_file, binary=False, no_header=True)
        return np.mean(model.vectors, axis=0)
    except Exception as e:
        sys.stderr.write('File %s caused an error: %s.\n' % (mittens_file, str(e)))


def self_centroid(files, num_files, embeddings_dir, time_type):
    """
    Main workhorse function for calculating embedding centroid for each person
    Parameters
    ----------
    files : list of str
        Embedding files to process
    num_files : int
        Total number of files to process, used to keep track of progress
    embeddings_dir : str
        Directory in which embedding files reside
    time_type : str
        The level at which we are computing obeservations for, one of year or quarter
    Return
    ------
    tuple
        3-tuple of dictionaries mapping usr and optional timekeys to within-person embedding similarities
    """
    usr_time2distances = defaultdict(list)
    pool = multiprocessing.Pool(processes = num_cores)
    results = {}
    for i, file in enumerate(files, 1):
        usr, timekey = extract_variables_from_file(file)
        if timekey:
            results[(usr, timekey)] = pool.apply_async(process_single_embedding_file, args=(i, num_files, embeddings_dir, file, ))
    pool.close()
    pool.join()
    for key, r in results.items():
        usr, timekey = key
        curr_row = r.get()
        # Empty if errored out
        if curr_row.any() and timekey:
            if time_type == 'year' and len(timekey) == 4:
                usr_time2distances[(usr, time_key)] = curr_row
            if time_type == 'quarter' and len(timekey) == 6:
                usr_time2distances[(usr, timekey)] = curr_row
    return usr_time2distances


def read_embeddings(embeddings_dir, time_type, test_mode=False):
    """
    Reads all files to calculate person-time embedding centroids
    Parameters
    ----------
    embeddings_dir : str
        Directory where all embedding files exist
    time_type : str
        One of quarter or year
    test_mode : bool, optional
        If testing, reduce number of files to process
    Returns
    -------
    tuple
        User, annual, and quarter level dataframes that include similarity between i and we
    """
    files = os.listdir(embeddings_dir)
    return self_centroid(files, len(files), embeddings_dir, time_type)

def fill_demographics():
    """
    This version differs from that in coco_network.py as this version allows for changes in department and other
    demographic variables over time.
    """
    user_qualtrics_df = pd.read_csv(user_qualtrics_file)[['Email', 'UID']]
    user_qualtrics_df.set_index('UID', inplace=True)
    uid2email = user_qualtrics_df['Email'].to_dict()
    
    hr_df_quarterly = pd.read_csv(long_hr_file)
    hr_df_quarterly['user_id'] = hr_df_quarterly.apply(lambda row : email2user_id.get(uid2email.get(row['uid'])), axis=1)
    hr_df = hr_df_quarterly[['user_id', 'race']].drop_duplicates().set_index('user_id')
    hr_df_quarterly.set_index(['user_id', 'quarter'], inplace=True)
    
    global userid2race, userid_quarter2age, userid_quarter2age, userid_quarter2cohort_year, userid_quarter2department, userid_quarter2title, userid_quarter2location, userid_quarter2country
    userid2race = hr_df['race'].apply(lambda s : np.nan if s == 'Missing' else s).dropna().to_dict()
    userid_quarter2age = hr_df_quarterly.apply(lambda row : row['year'] - row['year_of_birth'], axis=1).dropna().to_dict()
    userid_quarter2cohort_year = hr_df_quarterly.apply(lambda row : pd.to_datetime(row['GROUP DOJ'], format='%d-%b-%y').year, axis=1).dropna().to_dict()
    userid_quarter2department = hr_df_quarterly['department'].dropna().apply(lambda val : val.lower()).to_dict()
    userid_quarter2title = hr_df_quarterly['job_title'].dropna().apply(lambda val : val.lower()).to_dict()
    userid_quarter2location = hr_df_quarterly['work_location'].dropna().apply(lambda val : val.lower()).to_dict()
    userid_quarter2country = hr_df_quarterly['work_country'].dropna().apply(lambda val : val.lower()).to_dict()
    return

def compare_attributes(x, y, comp_type, window_size=None):
    if comp_type == 'equality':
        if x == y:
            return 'yes'
        return 'no'
    elif comp_type == 'window':
        if (x-y) <= window_size and (x-y) >= -1*window_size:
            return 'yes'
        return 'no'
    elif comp_type == 'binary_race_equality':
        if x == 'White' and y != 'White':
            return 'no'
        return 'yes'

def count_prop(group):
    return group.count('yes') / len(group) if len(group) > 0 else np.nan

def generate_network_composition(G_directed, timekey):
    """
    Generate ingroup proportion for directed Graph based on various attributes
    Parameter
    ---------
    G_directed : NetworkX graph
    timekey : str
        A string that represents the time period currently being computed, e.g., 2019Q3
    Returns
    -------
    tuple
        A list of dictionaries matching node (i.e., user ids) to various custom network variables
        node2*_ingroup : the proportion of one's network that consists of one's ingroup members as defined by *
        node2embeddedness : the degree to which one's local network is embedded in the broader organization as opposed to locally isolated
        node2cf : cultural fit as measured by the JS distance between the LIWC distribution of node's sent emails and that of node's neighbors
        node2peer_cf : group cultural fit as measured by the average pairwise JS distance in one's network, excluding oneself
        node2cossim : semantic similarity as measured by the cosine similarity between one's centroid and the average of the centroid of all of one's neighbors
        node2peer_cossim : group semantic similarity as measured by the pairwise cosine similarities in one's network, excluding oneself
    """
    (node2race_ingroup, node2bin_race_ingroup, node2age_ingroup, node2cohort_ingroup, node2department_ingroup, node2title_ingroup, node2location_ingroup, node2embeddedness,
        node2cf, node2peer_cf, node2cossim, node2peer_cossim) = [defaultdict(lambda : None) for _ in range(12)]
    # used for caching results that have already been computed
    u_v2distance, u_v2cossim = {}, {}
    for u in G_directed:
        if user_filter and (userid_quarter2country.get((u, timekey))) == "india":
            continue

        local_network = set(G_directed.neighbors(u))
        local_network.add(u)
        race_ingroup, race_binary, age_ingroup, cohort_ingroup, department_ingroup, title_ingroup, location_ingroup = [[] for _ in range(7)]
        within_cluster, without_cluster = 0, 0
        all_cf, all_centroid, all_cossim, num_centroid = {}, np.array([0.0]*50), list(), 0

        race = userid2race.get(u)
        age = userid_quarter2age.get((u, timekey))
        year = userid_quarter2cohort_year.get((u, timekey))
        department = userid_quarter2department.get((u, timekey))
        title = userid_quarter2title.get((u, timekey))
        location = userid_quarter2location.get((u, timekey))

        # As G is a directed network, u's network only includes those who u has sent an email to
        for v in G_directed.neighbors(u):
            v_dist = usr_quarter2dist[(v, timekey)]
            v_centroid = usr_quarter2centroid[(v, timekey)]
            
            if len(v_centroid) > 0 :
                all_centroid += v_centroid
                num_centroid += 1

            other_race = userid2race.get(v)
            if race and other_race:
                race_ingroup.append(compare_attributes(race, other_race, 'equality'))
                race_binary.append(compare_attributes(race, other_race, 'binary_race_equality'))
            other_age = userid_quarter2age.get((v, timekey))
            if age and other_age:
                age_ingroup.append(compare_attributes(age, other_age, 'window', 5))
            other_year = userid_quarter2cohort_year.get((v, timekey))
            if year and other_year:
                cohort_ingroup.append(compare_attributes(year, other_year, 'equality'))
            other_department = userid_quarter2department.get((v, timekey))
            if department and other_department:
                department_ingroup.append(compare_attributes(department, other_department, 'equality'))
            other_title = userid_quarter2title.get((v, timekey))
            if title and other_title:
                title_ingroup.append(compare_attributes(title, other_title, 'equality'))
            other_location = userid_quarter2location.get((v, timekey))
            if location and other_location:
                location_ingroup.append(compare_attributes(location, other_location, 'equality'))
                
            for z in G_directed.neighbors(v):
                if z in local_network:
                    within_cluster += 1
                else:
                    without_cluster += 1

            if compute_pair_distances:
                sys.stderr.write("Computing pairwise distances for %s.\n" % (timekey))
                # computing pair-wise distance measures
                for w in G_directed.neighbors(u):
                    # avoid double counting
                    if (v != w) and (v, w) not in all_cf and (w, v) not in all_cf:
                        distance = u_v2distance.get((v, w)) or u_v2distance.get((w, v))
                        if not distance:
                            distance = jensen_shannon(v_dist, usr_quarter2dist[(w, timekey)])
                            u_v2distance[(v, w)] = distance
                        if distance: all_cf[(v, w)] = distance
                        cossim = u_v2cossim.get((v, w)) or u_v2cossim.get((w, v))
                        if not cossim:
                            cossim = cossim_with_none(v_centroid, usr_quarter2centroid[(w, timekey)], 'dense')
                            u_v2cossim[(v, w)] = cossim
                        if cossim: all_cossim.append(cossim)

        node2race_ingroup[u] = count_prop(race_ingroup)
        node2bin_race_ingroup[u] = count_prop(race_binary)
        node2age_ingroup[u] = count_prop(age_ingroup)
        node2cohort_ingroup[u] = count_prop(cohort_ingroup)
        node2department_ingroup[u] = count_prop(department_ingroup)
        node2title_ingroup[u] = count_prop(title_ingroup)
        node2location_ingroup[u] = count_prop(location_ingroup)

        # if 1, no within_cluster ties, if -1, all within-cluster ties
        node2embeddedness[u] = (without_cluster-within_cluster)/(without_cluster+within_cluster)

        node2cf[u] = js2cf(jensen_shannon(usr_quarter2dist[(u, timekey)], node_term_count_distribution(G_directed.neighbors(u), usr_quarter2liwc, timekey)))
        node2cossim[u] = cossim_with_none(usr_quarter2centroid[(u, timekey)], all_centroid/num_centroid, 'dense')
        if compute_pair_distances:
            node2peer_cf[u] = js2cf(mean(all_cf.values())) if len(all_cf) > 0 else np.nan
            node2peer_cossim[u] = mean(all_cossim) if len(all_cossim) > 0 else np.nan
        

    return (node2race_ingroup, node2bin_race_ingroup, node2age_ingroup, node2cohort_ingroup, node2department_ingroup, node2title_ingroup, node2location_ingroup, 
        node2embeddedness, node2cf, node2peer_cf, node2cossim, node2peer_cossim)

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
    weighted_degree = G_directed.degree(weight='weight')
    unweighted_degree = G_directed.degree()
    G_undirected = G_directed.to_undirected()
    clique = nx.node_clique_number(G_undirected)
    directed_core = nx.core_number(G_directed)
    undirected_core = nx.core_number(G_undirected)
    between = nx.betweenness_centrality(G_directed, weight=None)
    close = nx.closeness_centrality(G_directed, distance=None)
    custom = generate_network_composition(G_directed, timekey)

    for n in G_directed:
        row = ([weighted_degree[n], unweighted_degree[n], clique[n], directed_core[n], undirected_core[n], between[n], close[n]])
        for l in custom:
            row.append(l[n])
        network_measures[(n, timekey)] = row

    sys.stderr.write("Finished network processing for %s at %s.\n" % (timekey, datetime.now()))
    return dict(network_measures)

def time_edges_to_df(time_edges, weighted=False, test_mode=False):
    """
    Calculates network measures using edge lists
    Parameters
    ----------
    time_edges : dict
        A dictionary that maps quarters (quarters only) to a list of edges belonging to that time period
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
        time_edges = {quarter:edges for quarter, edges in time_edges.items() if len(edges) > 5}
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
    cols = ['weighted_degree', 'unweighted_degree', 'clique', 'directed_core', 'undirected_core','unweighted_betweenness', 'closeness',
    'exact_race_ingroup', 'binary_race_ingroup', 'age_ingroup', 'cohort_ingroup', 'department_ingroup', 'title_ingroup', 'location_ingroup',
    'network_embeddedness', 'ego_cf', 'peer_cf', 'ego_cossim', 'peer_cossim']
    df = dict_to_df(network_measures, cols, index_name=['user_id', 'quarter'])
    return df.round(5)

def extract_network_measures(test_mode=False, weighted_mode='both'):
    """
    Main workhorse function for computing netwrork measures and writing them to file. Note that this function
    only computes measures quarterly.
    Parameters
    ----------
    test_mode : bool, optional
        If testing, modify file_name to include flags for testing in final output file name
    weighted_mode : bool, optional
        Whether to build a weighted or unweighted network. If weighted, email frequency betweeen the two parties are used
        as measure of strength.
    """
    sys.stderr.write("Computing edges at %s.\n" % str(datetime.now()))
    edges = get_quarterly_edges(test_mode)
    fill_demographics()
    sys.stderr.write("Calculating network measures at %s.\n" % str(datetime.now()))
    file_name = 'coco_network_supplemented'
    if user_filter:
        file_name += '_filtered'
    if not compute_pair_distances:
        file_name += '_no_pairwise'
    if test_mode:
        file_name += "_test"
    
    if weighted_mode == 'weighted' or weighted_mode == 'both':
        df = time_edges_to_df(edges, True, test_mode)
        existing_df = pd.read_csv(os.path.join(output_dir, "network_quarterly_internal_weighted.csv"))
        df = df.reset_index()
        df = existing_df.merge(df, on=['user_id', 'quarter'])
        df.to_csv(os.path.join(output_dir, file_name+"_weighted.csv"), index=False)
        sys.stderr.write("Finished outputting weighted network measures at %s.\n" % str(datetime.now()))

    if weighted_mode == 'unweighted' or weighted_mode == 'both':
        df = time_edges_to_df(edges, False, test_mode)
        existing_df = pd.read_csv(os.path.join(output_dir, "network_quarterly_internal_unweighted.csv"))
        df = df.reset_index()
        df = existing_df.merge(df, on=['user_id', 'quarter'])
        df.to_csv(os.path.join(output_dir, file_name+"_unweighted.csv"), index=False)
        sys.stderr.write("Finished outputting unweighted network measures at %s.\n" % str(datetime.now()))
    return

if __name__ == '__main__':
    starttime = datetime.now()
    test_mode, weighted_mode = False, 'weighted'
    try:
        test_mode = sys.argv[1].lower() == 'test'
        weighted_mode = sys.argv[2].lower()
    except IndexError as error:
        pass

    sys.stderr.write('Reading embeddings at %s.\n' % datetime.now())
    if compute_centroid:
        usr_quarter2centroid = read_embeddings(embeddings_dir, 'quarter', test_mode)
    else:
        d = pd.read_csv(centroid_file).set_index(['user_id', 'quarter']).to_dict('index')
        usr_quarter2centroid = defaultdict(list)
        for key, row in d.items():
            usr_quarter2centroid[key] = np.array(list(row.values()))
        
    sys.stderr.write('Generating Network Measures at %s.\n' % datetime.now())
    extract_network_measures(test_mode, weighted_mode)
    
    sys.stderr.write("Finished running at %s, with a duration of %s.\n"
        % (str(datetime.now()), str(datetime.now() - starttime)))
