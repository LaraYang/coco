#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    3a_output_embeddings.py test|actual
If a Negative Dimension Error occurs, check to see if there are empty embeddings
Three types of results are currently computed
1) i to we + i-cluster to we-cluster
2) i to company i, we to company we, i-cluster to company i-cluster, we-cluster to company we-cluster
3) i to average i, we to average we, i-cluster to average i-cluster, we-cluster to average we-cluster

"""
import os
import sys
import multiprocessing
from collections import defaultdict
import pandas as pd
import numpy as np
import csv
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.matutils import cossim, any2sparse
from utils import *
import re
import random
from statistics import mean 
import ujson as json


embedding_dim = 50
mincount = 50
ling_thres = 0.9
home_dir = "/zfs/projects/faculty/amirgo-identification/"
email_dir = os.path.join(home_dir, "email_data/")
mittens_dir = os.path.join(home_dir, "mittens")
utils_dir = os.path.join(mittens_dir, "utils")
embeddings_dir = os.path.join(mittens_dir, "embeddings_high_prob_eng_{}_{}d_mincount{}".format(str(ling_thres).replace(".", ""), embedding_dim, mincount))

email_file = os.path.join(email_dir, 'MessagesHashed.jsonl')
users_file = os.path.join(email_dir, 'Users.json')
activity_file = os.path.join(email_dir, 'Activities.json')
survey_dir = os.path.join(home_dir, "survey_hr_data")
user_qualtrics_file = os.path.join(survey_dir, "UsersQualtrics.csv")
perf_percentage = os.path.join(survey_dir, "perf_rating_percentages.csv")
perf_likert = os.path.join(survey_dir, "perf_rating_likert.csv")

analyses_data_dir = "/zfs/projects/faculty/amirgo-transfer/spacespace/spacespace/Coco/analyses_data/"
survey_filename = os.path.join(analyses_data_dir, "preprocessed_survey_hr.csv")
company_embeddings_filename = "/zfs/projects/faculty/amirgo-transfer/spacespace/spacespace/Coco/Embed/GloVe-master/vectors_high_prob_eng_{}_{}d.txt".format(str(ling_thres).replace(".", ""), embedding_dim)

tmp_dir = os.path.join(mittens_dir, "tmp")
output_dir = os.path.join(home_dir, "coco_email_idtf_data")

user_output_filename = os.path.join(output_dir, "embeddings_high_prob_eng_{}_users_{}d_mincount{}.csv".format(str(ling_thres).replace(".", ""), embedding_dim, mincount))
annual_output_filename = os.path.join(output_dir, "embeddings_high_prob_eng_{}_annual_{}d_mincount{}.csv".format(str(ling_thres).replace(".", ""), embedding_dim, mincount))
quarterly_output_filename = os.path.join(output_dir, "embeddings_high_prob_eng_{}_quarterly_{}d_mincount{}.csv".format(str(ling_thres).replace(".", ""), embedding_dim, mincount))

year_colname, quarter_colname = 'year', 'quarter'
hash2word = {
    '09f83385': 'mine',
    '20019fa4': 'i',
    '20b60145': 'us',
    '28969cb1': 'them',
    '3828d3d2': 'me',
    '4dd6d391': 'their',
    '5b4e27db': 'my',
    '64a505fc': 'ourselves',
    '6935bb23': 'ours',
    '6f75419e': 'myself',
    '86df0c8d': 'themselves',
    'a7383e72': 'we',
    'a9193217': 'theirs',
    'b72a9dd7': 'our',
     'fd0ccf1c': 'they'}
word2hash = {v:k for k, v in hash2word.items()}
pronouns = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']
single_pronouns = ['i', 'we']
i_index, we_index = 0, 5
hash_pronouns = [word2hash[p] for p in pronouns]
hash_single_pronouns = [word2hash[p] for p in single_pronouns]

file_name_re = re.compile("5([a-z0-9]+)_(2020(Q3)?_)?internal.txt")
num_cores = 14

domain_hash = {
    'collabera.com':                     '509c8f6b1127bceefd418c023533d653',
    'collaberainc.mail.onmicrosoft.com': 'ec5b67548b6ec06f1234d198efec741e',
    'collaberainc.onmicrosoft.com':      '86160680578ee9258f097a67a5f25af9',
    'collaberasp.com':                   '6bf3934d19f1acf5b9295b63e0e7f66e',
    'g-c-i.com':                         '3444d1f7d5e46443080f2d069e41a10c'}
collabera_hashes = set([v for k, v in domain_hash.items()])


#########################################################################
######### Functions for Loading Raw Counts as Control Variables #########
#########################################################################

def read_raw_counts(activity_file, email_file):
    """
    The main workhorse function for obtaining raw message and token counts as control variables.
    Parameters
    ----------
    activity_file : str
        The full filepath that contains all email metadata, where each line is a JSON object that represents one email
    email_file : str
        The full filepath that contains all email content, where each line is a JSON object that represents one email
    Returns
    -------
    tuple
        A tuple of user-level, annual, and quarterly dataframes
    """
    usr2counts, usr_year2counts, usr_quarter2counts = defaultdict(lambda : [0, 0]), defaultdict(lambda : [0, 0]), defaultdict(lambda :[0, 0])
    sid2activity = {}
    cols = ['num_tokens', 'num_messages']
    tok_count_index, msg_count_index = 0, 1
    with open(activity_file, encoding='utf-8') as f:
        for line in f:
            activity = json.loads(line)
            sid2activity[activity['MailSummarySid']] = activity

    with open(email_file, encoding='utf-8') as f:
        for line in f:
            email = json.loads(line)
            lang = email['l']
            if len(email['hb']) > 0 and lang[0] == "__label__en" and lang[1] > ling_thres:
                activity = sid2activity[email['sid']]
                user = activity['UserId']
                year = to_year(activity['ActivityCreatedAt'], format='str')
                quarter = to_quarter(activity['ActivityCreatedAt'], format='str')
                num_toks = len(email['hb'].replace('\n', ' ').replace("SENT_END", "").strip().split())
                activity = sid2activity[email['sid']]
                recipients = get_recipients(activity)
                pure_internal = True
                for r in recipients:
                    domain = r.split('@')[1]
                    if domain not in collabera_hashes:
                        pure_internal = False
                        break
                if not pure_internal:
                    continue
                usr2counts[user][tok_count_index] += num_toks
                usr2counts[user][msg_count_index] += 1
                usr_year2counts[(user, year)][tok_count_index] += num_toks
                usr_year2counts[(user, year)][msg_count_index] += 1
                usr_quarter2counts[(user, quarter)][tok_count_index] += num_toks
                usr_quarter2counts[(user, quarter)][msg_count_index] += 1

    usr2counts_df = dict_to_df(usr2counts, cols, index_name=['user_id'])
    usr_year2counts_df = dict_to_df(usr_year2counts, cols, index_name=['user_id', year_colname])
    usr_quarter2counts_df = dict_to_df(usr_quarter2counts, cols, index_name=['user_id', quarter_colname])
    return (usr2counts_df, usr_year2counts_df, usr_quarter2counts_df)

#########################################################################
#### Functions for Measuring Within-Person Similarities in Embeddings ###
#########################################################################

def embeddings_similarities(model):
    """
    Returns the embeddings of i, we, centroid of i-words, centroid of we-words, and their respective cosine similarities
    Parameters
    ----------
    model : gensim.models.Word2Vec
        Model that stores the embeddings for each word
    Returns
    -------
    embeds : list
        A list of embeddings (vectors) and similarities (float)
    """
    i = model.wv[hash_single_pronouns[0]] if hash_single_pronouns[0] in model.vocab else None
    we = model.wv[hash_single_pronouns[1]] if hash_single_pronouns[1] in model.vocab else None
    i_cluster = [model.wv[word] for word in hash_pronouns[i_index:we_index] if word in model.vocab]
    i_cluster = None if len(i_cluster) == 0 else np.mean(i_cluster, axis=0)
    we_cluster = [model.wv[word] for word in hash_pronouns[we_index:] if word in model.vocab]
    we_cluster = None if len(we_cluster) == 0 else np.mean(we_cluster, axis=0)
    embeds = ([i, we] + [word_similarity(model, hash_pronouns[i_index+i], hash_pronouns[we_index+i]) for i in range(5)] +
        [i_cluster, we_cluster, word_similarity(model, hash_pronouns[i_index:we_index], hash_pronouns[we_index:])])
    symmetric_i_words, symmetric_we_words = [], []
    for i in range(len(hash_pronouns)-we_index):
        if hash_pronouns[i] in model.vocab and hash_pronouns[i+we_index] in model.vocab:
            symmetric_i_words.append(hash_pronouns[i])
            symmetric_we_words.append(hash_pronouns[i+we_index])
    if len(symmetric_i_words) > 0:
        embeds.append(model.n_similarity(symmetric_i_words, symmetric_we_words))
    return embeds

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
    embeds : list
        A list of embeddings and similarities. Embeddings are used for calculating between-person similarities in downstream functions.
    """
    mittens_file = os.path.join(embeddings_dir, file)
    sys.stderr.write("Processing \t%d/%d -'%s', at %s.\n" % (i, num_files, mittens_file, datetime.now()))
    # chopping off the file extension in filename
    tmp_mittens = os.path.join(tmp_dir, file[0:-4] + "_word2vec.txt")
    try:
        word2vec_mittens_file = get_tmpfile(tmp_mittens)
        glove2word2vec(mittens_file, word2vec_mittens_file)
        model = KeyedVectors.load_word2vec_format(word2vec_mittens_file)
        embeds = embeddings_similarities(model)
        return embeds
    except Exception as e:
        sys.stderr.write('File %s caused an error: %s.\n' % (mittens_file, str(e)))
   
def self_similarities(files, num_files, embeddings_dir):
    """
    Main workhorse function for calculating within-person similarities. Compares an individual's i embedding to we embedding, using both
    i and we's embedding only and centroid of i-words and we-words. Indices used in this file relies on knowledge of the naming convention of underlying embedding files.
    Parameters
    ----------
    files : list of str
        Embedding files to process
    num_files : int
        Total number of files to process, used to keep track of progress
    embeddings_dir : str
        Directory in which embedding files reside
    Return
    ------
    tuple
        3-tuple of dictionaries mapping usr and optional timekeys to within-person embedding similarities
    """
    usr2distances, usr_year2distances, usr_quarter2distances = defaultdict(list), defaultdict(list), defaultdict(list)
    pool = multiprocessing.Pool(processes = num_cores)
    results = {}
    for i, file in enumerate(files, 1):
        usr, time_key = extract_variables_from_file(file)
        results[(usr, time_key)] = pool.apply_async(process_single_embedding_file, args=(i, num_files, embeddings_dir, file, ))
    pool.close()
    pool.join()
    for key, r in results.items():
        usr, time_key = key
        curr_row = r.get()
        # Empty if errored out
        if curr_row:
            if time_key:
                if len(time_key) == 4:
                    usr_year2distances[(usr, time_key)] = curr_row
                elif len(time_key) == 6:
                    usr_quarter2distances[(usr, time_key)] = curr_row
                else:
                    sys.stderr.write('Embedding file format does not conform to expectations. Extracted time key %s for user %s.\n' % (time_key, usr)) 
            else:
                usr2distances[(usr)] = curr_row
    return (usr2distances, usr_year2distances, usr_quarter2distances)

#########################################################################
### Functions for Measuring Between-Person Similarities in Embeddings ###
#########################################################################
def pairwise_cossim(df, i_col, we_col, reference_tag='', reference=False, reference_group=None, anon_ids=None, vec_format='sparse'):
    """
    Calculating pairwise cosine similarities between every i-embedding to every other i-embedding, and every we-embedding to every other we-embedding
    """
    col1, col2 = 'avg_i_i' + reference_tag, 'avg_we_we' + reference_tag
    df[col1] = calculate_pairwise_cossim(df[i_col], reference=reference, reference_group=reference_group, anon_ids=anon_ids, vec_format=vec_format)
    df[col2] = calculate_pairwise_cossim(df[we_col], reference=reference, reference_group=reference_group, anon_ids=anon_ids, vec_format=vec_format)
    return df

def sparsify(df, to_sparse, sparse):
    assert len(to_sparse) == len(sparse)
    for i in range(len(to_sparse)):
        df[sparse[i]] = df[to_sparse[i]].apply(lambda x : any2sparse(x) if (x is not None and np.isfinite(x).all()) else None)
    return df
    
def cossimify(df, names, vec_format='sparse'):
    """
    Parameters
    ----------
    df : pd.DataFrame
    names : list
        A list of tuples, where the first element is the first column name, second element is the second column name, and third element is the result column name
    Returns
    -------
    df : pd.DataFrame
        The original dataframe with the cosine similarity columns appended
    """
    for tup in names:
        df[tup[2]] = df.apply(lambda row : cossim_with_none(row[tup[0]], row[tup[1]], vec_format), axis=1)
    return df

def self_other_similarities(df, company_embeddings, company_cluster_embeddings, panel_data):
    """
    Main workhorse function for calculating between-person similarities. Compares an individual's embeddings to
    the average embeddings of everyone else in the company, as well as GloVe embeddings built on the entire company's corpus.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by self_similarities that includes both embeddings and embedding similarities at the individual level
    company_embeddings : 2-tuple of numpy array
        Company embeddings of "i" and "we"
    company_cluster_embeddings : 2-tuple of numpy array
        Company embeddings of centroid of "i" words and centroid of "we" words
    panel_data : bool
        Indicates whether df is in a panel data format or a cross-sectional format. If in panel format,
        grouping by timekey is needed before averaging.
    Returns
    -------
    df : pd.DataFrame
        Dataframe that includes both within- and between-person similarities
    """
    if df is None or df.empty:
        return
    sparse_company_embeddings = [any2sparse(company_embeddings[0]), any2sparse(company_embeddings[1])]
    sparse_company_cluster_embeddings = [any2sparse(company_cluster_embeddings[0]), any2sparse(company_cluster_embeddings[1])]

    # converts np.array or scipy array to a sparse vector format used by gensim
    # any doesn't mean anything -- Python lists are not accepted by this function
    # sparse vectors are lists whose first elements are indices and second elements are numbers
    # the indices correspond to indices of the original vector fed into any2sparse, where zeroes were retained
    # sparse vectors through out all the zeroes in the actual vector to save space
    # are vector indices of the dense vector 
    df['i_sparse'] = df['i_embed'].apply(lambda x : any2sparse(x) if not isnull_wrapper(x) else None)
    df['we_sparse'] = df['we_embed'].apply(lambda x: any2sparse(x) if not isnull_wrapper(x) else None)
    df['i_cluster_sparse'] = df['i_cluster'].apply(lambda x : any2sparse(x) if not isnull_wrapper(x) else None)
    df['we_cluster_sparse'] = df['we_cluster'].apply(lambda x : any2sparse(x) if not isnull_wrapper(x) else None)

    if not panel_data:
        i_mean = vector_mean(df['i_embed'])
        df['i_embed_avg'] = df.apply(lambda x : i_mean, axis=1)
        we_mean = vector_mean(df['we_embed'])
        df['we_embed_avg'] = df.apply(lambda x : we_mean, axis=1)
        i_mean = vector_mean(df['i_cluster'])
        df['i_cluster_avg'] = df.apply(lambda x : i_mean, axis=1)
        we_mean = vector_mean(df['we_cluster'])
        df['we_cluster_avg'] = df.apply(lambda x : we_mean, axis=1)
        df = pairwise_cossim(df, 'i_sparse', 'we_sparse', vec_format='sparse')
        df = pairwise_cossim(df, 'i_cluster_sparse', 'we_cluster_sparse', reference_tag='_cluster', vec_format='sparse')
    else:
        #  If i_embed or we_embed is not defined for any anon_id during this period,
        # then vector_mean will return np.nan
        df = df.join(df['i_embed'].groupby(level=1).apply(vector_mean), rsuffix='_avg') 
        df = df.join(df['we_embed'].groupby(level=1).apply(vector_mean), rsuffix='_avg')
        df = df.join(df['i_cluster'].groupby(level=1).apply(vector_mean), rsuffix='_avg') 
        df = df.join(df['we_cluster'].groupby(level=1).apply(vector_mean), rsuffix='_avg')
        new_df = pd.DataFrame()
        for time_chunk, time_df in df.groupby(level=1):
            time_df = pairwise_cossim(time_df, 'i_sparse', 'we_sparse', vec_format='sparse')
            time_df = pairwise_cossim(time_df, 'i_cluster_sparse', 'we_cluster_sparse', reference_tag='_cluster', vec_format='sparse')
            new_df = new_df.append(time_df)
        df = new_df
    # Filtering out Nones and np.nan (who has class float)
    df = sparsify(df, ['i_embed_avg', 'we_embed_avg', 'i_cluster_avg', 'we_cluster_avg'],
        ['i_avg_sparse', 'we_avg_sparse', 'i_cluster_avg_sparse', 'we_cluster_avg_sparse'])
    
    # These averages are already defined based on time periods when we are running using panel data due to the inherent structure of the data
    df = cossimify(df,
        [('i_sparse', 'i_avg_sparse', 'i_avg_i'), ('we_sparse', 'we_avg_sparse', 'we_avg_we'),
        ('i_cluster_sparse', 'i_cluster_avg_sparse', 'i_avg_i_cluster'), ('we_cluster_sparse', 'we_cluster_avg_sparse', 'we_avg_we_cluster'),
        ('i_sparse', 'we_avg_sparse', 'i_avg_we'), ('i_cluster_sparse', 'we_cluster_avg_sparse', 'i_avg_we_cluster')], vec_format='sparse')

    df['i_company_i'] = df.apply(lambda row : cossim_with_none(row['i_sparse'], sparse_company_embeddings[0], vec_format='sparse'), axis=1) 
    df['we_company_we'] = df.apply(lambda row : cossim_with_none(row['we_sparse'], sparse_company_embeddings[1], vec_format='sparse'), axis=1) 
    df['i_company_i_cluster'] = df.apply(lambda row : cossim_with_none(row['i_cluster_sparse'], sparse_company_cluster_embeddings[0], vec_format='sparse'), axis=1) 
    df['we_company_we_cluster'] = df.apply(lambda row : cossim_with_none(row['we_cluster_sparse'], sparse_company_cluster_embeddings[1], vec_format='sparse'), axis=1) 
    df['i_company_we'] = df.apply(lambda row : cossim_with_none(row['i_sparse'], sparse_company_embeddings[1], vec_format='sparse'), axis=1) 
    df['i_company_we_cluster'] = df.apply(lambda row : cossim_with_none(row['i_cluster_sparse'], sparse_company_cluster_embeddings[1], vec_format='sparse'), axis=1) 
    return df.round(5)

def compare_internal_external(df):
    df = cossimify(df,
        [('i_sparse_internal', 'we_sparse_external', 'i_int_we_ext'),
        ('i_cluster_sparse_internal', 'we_cluster_sparse_external', 'i_int_we_ext_cluster'),
        ('i_sparse_internal', 'we_avg_sparse_external', 'i_int_we_avg_ext'),
        ('i_cluster_sparse_internal', 'we_cluster_avg_sparse_external', 'i_int_we_avg_ext_cluster')], vec_format='sparse')

    for post in ['_internal', '_external']:
        cols = ['i_embed', 'we_embed', 'i_sparse', 'we_sparse', 'i_embed_avg', 'we_embed_avg', 'i_avg_sparse', 'we_avg_sparse',
        'i_cluster', 'we_cluster', 'i_cluster_sparse', 'we_cluster_sparse', 'i_cluster_avg', 'we_cluster_avg', 'i_cluster_avg_sparse', 'we_cluster_avg_sparse']
        cols = [c+post for c in cols]
        df.drop(cols, axis=1, inplace=True)
    return df

def reading_embeddings(embeddings_dir, company_embeddings, company_cluster_embeddings, test_mode=False):
    """
    Calculates embedding similarities within-person and between-person
    Parameters
    ----------
    embeddings_dir : str
        Directory where all embedding files exist
    company_embeddings : tuple of numpy array
        Embeddings of "i" and "we" in the whole company email corpus
    company_cluster_embeddings : tuple of numpy array
        Embeddings of average of all "i" words and average of all "we" words in the whole company email corpus
    test_mode : bool, optional
        If testing, reduce number of files to process
    Returns
    -------
    tuple
        User, annual, and quarter level dataframes that include both within- and between-person embedding similarities
    """
    all_files = os.listdir(embeddings_dir)
    if test_mode: all_files = [all_files[random.randint(0, len(all_files)-1)] for _ in range(len(all_files)//50)]
    internal_re = re.compile(".+_internal.txt")
    external_re = re.compile(".+_external.txt")
    internal_files, external_files = [], []
    for f in all_files:
        if re.match(internal_re, f):
            internal_files.append(f)
        elif re.match(external_re, f):
            external_files.append(f)

    result_dfs = []
    for i, files in enumerate([internal_files, external_files], 1):
        num_files = len(files)
        sys.stderr.write('Iteration %d: Calculate within-person similarities for %d files at %s.\n' % (i, num_files, str(datetime.now())))
        usr2distances, usr_year2distances, usr_quarter2distances = self_similarities(files, num_files, embeddings_dir)

        cols = ['i_embed', 'we_embed', 'i_we', 'me_us', 'my_our', 'mine_ours', 'myself_ourselves', 'i_cluster', 'we_cluster', 'i_we_cluster', 'i_we_symmetric']
        usr2distances_df = dict_to_df(usr2distances, cols, index_name=['user_id'])
        usr_year2distances_df = dict_to_df(usr_year2distances, cols, index_name=['user_id', year_colname])
        usr_quarter2distances_df = dict_to_df(usr_quarter2distances, cols, index_name=['user_id', quarter_colname])

        sys.stderr.write('Iteration %d: Calculate between-person similarities for %d files at %s.\n' % (i, num_files, str(datetime.now())))
        pool = multiprocessing.Pool(processes = num_cores)
        results = ([pool.apply_async(self_other_similarities, args=(df, company_embeddings, company_cluster_embeddings, panel,))
            for df, panel in [(usr2distances_df, False), (usr_year2distances_df, True), (usr_quarter2distances_df, True)]])
        pool.close()
        pool.join()
        result_dfs.append([r.get() for r in results])
        sys.stderr.write('Iteration %d: Successfully read and computed cosine similarities for %d embedding files at %s.\n' % (i, num_files, str(datetime.now())))  
        
    usr_df = result_dfs[0][0].join(result_dfs[1][0], lsuffix='_internal', rsuffix='_external', how='outer')
    usr_year_df = result_dfs[0][1].join(result_dfs[1][1], lsuffix='_internal', rsuffix='_external', how='outer')
    usr_quarter_df = result_dfs[0][2].join(result_dfs[1][2], lsuffix='_internal', rsuffix='_external', how='outer')

    usr_df = compare_internal_external(usr_df)
    usr_year_df = compare_internal_external(usr_year_df)
    usr_quarter_df = compare_internal_external(usr_quarter_df)
    return (usr_df, usr_year_df, usr_quarter_df)

if __name__ == '__main__':
    starttime = datetime.now()
    test = False
    try:
        test = sys.argv[1].lower() == 'test'
    except IndexError as error:
        pass
    if test:
        user_output_filename = os.path.join(output_dir, "test_embeddings_users.csv")
        annual_output_filename = os.path.join(output_dir, "test_embeddings_annual.csv")
        quarterly_output_filename = os.path.join(output_dir, "test_embeddings_quarterly.csv")
    for d in [output_dir, tmp_dir]:
        if not os.path.exists(d):
            os.mkdir(d)
    
    sys.stderr.write('Loading corpus counts at %s.\n' % datetime.now())
    usr2counts, usr2annual_counts, usr2quarterly_counts = read_raw_counts(activity_file, email_file)
    sys.stderr.write('Reading embeddings at %s.\n' % datetime.now())
    company_embeddings = extract_company_embedding(company_embeddings_filename, tmp_dir, hash_pronouns)
    company_cluster_embeddings = (vector_mean(pd.Series(company_embeddings[i_index:we_index])), vector_mean(pd.Series(company_embeddings[we_index:])))
    usr2measures, usr2annual_measures, usr2quarterly_measures = reading_embeddings(embeddings_dir, company_embeddings, company_cluster_embeddings, test)
    
    # different embedding files should be matched with the same hr file as hr data is not in panel format
    sys.stderr.write('Reading HR and Survey data at %s.\n' % datetime.now())
    hr_df = extract_hr_survey_df(survey_filename, user_qualtrics_file, users_file, perf_likert, perf_percentage)

    # could just merge and write to csv without calling another function
    sys.stderr.write('Outputting dataframe at %s.\n' % datetime.now())
    if usr2measures is not None: hr_df.join(usr2measures).join(usr2counts).to_csv(user_output_filename)
    if usr2annual_measures is not None: hr_df.join(usr2annual_measures).join(usr2annual_counts).to_csv(annual_output_filename)
    if usr2quarterly_measures is not None: hr_df.join(usr2quarterly_measures).join(usr2quarterly_counts).to_csv(quarterly_output_filename)
    
    sys.stderr.write("Finished outputting measures at %s, with a duration of %s.\n"
        % (str(datetime.now()), str(datetime.now() - starttime)))
