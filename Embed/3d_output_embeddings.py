#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    3d_output_embeddings.py test|actual
This file allows for supervised learning down the line where i-vectors, we-vectors, and the vector that represents i minus we, are used as features.
"""
import os
import sys
import multiprocessing
from collections import defaultdict, Counter
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
import traceback
from sklearn.decomposition import PCA

embedding_dim = 50
mincount = 150
home_dir = "/ifs/projects/amirgo-identification/"
email_dir = os.path.join(home_dir, "email_data/")
mittens_dir = os.path.join(home_dir, "mittens")
utils_dir = os.path.join(mittens_dir, "utils")
embeddings_dir = os.path.join(mittens_dir, "embeddings_{}d_mincount{}".format(embedding_dim, mincount))
email_file = os.path.join(email_dir, 'MessagesHashed.jsonl')
users_file = os.path.join(email_dir, 'Users.json')
activity_file = os.path.join(email_dir, 'Activities.json')
survey_dir = os.path.join(home_dir, "survey_hr_data")
user_qualtrics_file = os.path.join(survey_dir, "UsersQualtrics.csv")
perf_percentage = os.path.join(survey_dir, "perf_rating_percentages.csv")
perf_likert = os.path.join(survey_dir, "perf_rating_likert.csv")

analyses_data_dir = "/ifs/gsb/amirgo/spacespace/spacespace/Coco/analyses_data/"
survey_filename = os.path.join(analyses_data_dir, "preprocessed_survey_hr.csv")
company_embeddings_filename = "/ifs/gsb/amirgo/spacespace/spacespace/Coco/Embed/GloVe-master/vectors_{}d.txt".format(embedding_dim)

tmp_dir = os.path.join(mittens_dir, "tmp")
output_dir = os.path.join(home_dir, "email_idtf_data")

user_output_filename = os.path.join(output_dir, "embeddings_users_vectors_{}d_mincount{}.csv".format(embedding_dim, mincount))
annual_output_filename = os.path.join(output_dir, "embeddings_annual_vectors_{}d_mincount{}.csv".format(embedding_dim, mincount))
quarterly_output_filename = os.path.join(output_dir, "embeddings_quarterly_vectors_{}d_mincount{}.csv".format(embedding_dim, mincount))

year_colname, quarter_colname = 'year', 'quarter'
hash2word = {
    '09f83385': 'mine', '20019fa4': 'i', '20b60145': 'us', '28969cb1': 'them', '3828d3d2': 'me', '4dd6d391': 'their', '5b4e27db': 'my',
    '64a505fc': 'ourselves', '6935bb23': 'ours', '6f75419e': 'myself', '86df0c8d': 'themselves', 'a7383e72': 'we', 'a9193217': 'theirs', 'b72a9dd7': 'our', 'fd0ccf1c': 'they', 
    'ce696289': 'home', 'b95eb14b': 'attached', '267430a0': 'good', '294fa7d1': 'collabera', '974811d0': 'pay', 'edbf568e': 'work', 'b71be0e8': 'team', '4c088971': 'great',
    'c74560f9': 'best', 'f18e6868': 'different', '1f4d7738': 'group', '255ddfcd': 'glad', 'aa829423': 'included', '17e1378b': 'money', '454ea538': 'salary', '311b8ad0': 'community',
    '3b75b927': 'happy', '9324aa22': 'organized', '63b8b7ea': 'bad', '643ce56f': 'responsive', 'f4732b84': 'enthusiastic', '2e32c475': 'competitive', 'b9625ccf': 'family',
    '900c73ff': 'unresponsive', 'cfe1bd08': 'income', '223deabb': 'worst', 'fa81b32a': 'pride', '1455e3bd': 'passionate', '9582e03b': 'awful', 'd9f0fe6c': 'promotion',
    'c40b5da1': 'excluded', 'cf9cb85a': 'ambitious', 'a0cb3a2b': 'sad', '8a4e04bd': 'honor', 'cafaa726': 'belong', '24cb6fe3': 'shame', 'b92208fc': 'disciplined', '68e0c9c9': 'undisciplined',
    '81bcf2f5': 'receptive', '8ca67680': 'disorganized', 'd22e4710': 'bitter', 'bf4db4c4': 'unenthusiastic', '8602bd25': 'dignity', '822f792d': 'detached', 'a7ca40f1': 'humiliation',
    '7911da73': 'noncompetitive', '627fcac3': 'dishonor', '84cadff4': 'unreceptive', '07ca39d6': 'lazy', '95a160e0': 'indifferent', '10a4d7ee': 'apathetic'}
word2hash = {v:k for k, v in hash2word.items()}
pronouns = ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']
single_pronouns = ['i', 'we']
i_index, we_index = 0, 5
hash_pronouns = [word2hash[p] for p in pronouns]
hash_single_pronouns = [word2hash[p] for p in single_pronouns]

num_cores = 12
domain_hash = {
    'collabera.com':                     '509c8f6b1127bceefd418c023533d653',
    'collaberainc.mail.onmicrosoft.com': 'ec5b67548b6ec06f1234d198efec741e',
    'collaberainc.onmicrosoft.com':      '86160680578ee9258f097a67a5f25af9',
    'collaberasp.com':                   '6bf3934d19f1acf5b9295b63e0e7f66e',
    'g-c-i.com':                         '3444d1f7d5e46443080f2d069e41a10c'}
collabera_hashes = set([v for k, v in domain_hash.items()])

def read_raw_email(activity_file, email_file, name2dims, test_mode):
    """
    The main workhorse function for obtaining word frequency counts that might relevant for identification
    as well as computing average projection score of global vectors in raw emails.
    Parameters
    ----------
    activity_file : str
        The full filepath that contains all email metadata, where each line is a JSON object that represents one email
    email_file : str
        The full filepath that contains all email content, where each line is a JSON object that represents one email
    name2dims : dict
        Dictionary matching dimension names to actual 2-tuples of dimensions
    test_mode: bool
        If true, run a small number of files
    Returns
    -------
    tuple
        A tuple of user-level, annual, and quarterly dataframes
    """
    usr2counts, usr_year2counts, usr_quarter2counts = defaultdict(lambda : [0] * num_cols), defaultdict(lambda : [0] * num_cols), defaultdict(lambda :[0] * num_cols)
    sid2activity = {}
    with open(activity_file, encoding='utf-8') as f:
        for line in f:
            activity = json.loads(line)
            sid2activity[activity['MailSummarySid']] = activity

    with open(email_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if test_mode and i > 10000: break
            if (i+1) % 100000 == 0:
                sys.stderr.write('Processing email #%d.\n' % (i+1))
            email = json.loads(line)
            lang = email['l']
            if len(email['hb']) > 0 and lang[0] == "__label__en" and lang[1] > 0.5:
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
                user = activity['UserId']
                year = to_year(activity['ActivityCreatedAt'], format='str')
                quarter = to_quarter(activity['ActivityCreatedAt'], format='str')
        
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
    i_we_diff = None if i is None or we is None else we-i
    i_cluster = [model.wv[word] for word in hash_pronouns[i_index:we_index] if word in model.vocab]
    i_cluster = None if len(i_cluster) == 0 else np.mean(i_cluster, axis=0)
    we_cluster = [model.wv[word] for word in hash_pronouns[we_index:] if word in model.vocab]
    we_cluster = None if len(we_cluster) == 0 else np.mean(we_cluster, axis=0)
    i_we_cluster_diff = None if i_cluster is None or we_cluster is None else we_cluster-i_cluster
    embeds = ([i, we, i_we_diff, word_similarity(model, hash_single_pronouns[0], hash_single_pronouns[1]),
        i_cluster, we_cluster, i_we_cluster_diff, word_similarity(model, hash_pronouns[i_index:we_index], hash_pronouns[we_index:])])
    symmetric_i_words, symmetric_we_words = [], []
    for i in range(len(hash_pronouns)-we_index):
        if hash_pronouns[i] in model.vocab and hash_pronouns[i+we_index] in model.vocab:
            symmetric_i_words.append(hash_pronouns[i])
            symmetric_we_words.append(hash_pronouns[i+we_index])
    if len(symmetric_i_words) > 0:
        symmetric_we = np.mean([model.wv[word] for word in symmetric_we_words], axis=0)
        symmetric_i = np.mean([model.wv[word] for word in symmetric_i_words], axis=0)
        embeds.append(symmetric_we - symmetric_i)
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
        traceback.print_exc()
   
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

def reading_embeddings(embeddings_dir, test_mode=False):
    """
    Calculates embedding similarities within-person and between-person
    Parameters
    ----------
    embeddings_dir : str
        Directory where all embedding files exist
    test_mode : bool, optional
        If testing, reduce number of files to process
    Returns
    -------
    tuple
        User, annual, and quarter level dataframes that include within-person embedding vectors and difference vectors
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

    sys.stderr.write('Calculate within-person similarities for %d internal files at %s.\n' % (len(internal_files), str(datetime.now())))
    usr2distances, usr_year2distances, usr_quarter2distances = self_similarities(internal_files, len(internal_files), embeddings_dir)

    cols = ['i_embed', 'we_embed', 'i_we_diff', 'i_we', 'i_cluster', 'we_cluster', 'i_we_cluster_diff', 'i_we_cluster', 'i_we_symmetric_diff', 'i_we_symmetric']
    usr2distances_df = dict_to_df(usr2distances, cols, index_name=['user_id'])
    usr_year2distances_df = dict_to_df(usr_year2distances, cols, index_name=['user_id', year_colname])
    usr_quarter2distances_df = dict_to_df(usr_quarter2distances, cols, index_name=['user_id', quarter_colname])
    
    sys.stderr.write('Calculate within-person similarities for %d external files at %s.\n' % (len(external_files), str(datetime.now())))
    usr2distances, usr_year2distances, usr_quarter2distances = self_similarities(external_files, len(external_files), embeddings_dir)
    usr2distances_df_external = dict_to_df(usr2distances, cols, index_name=['user_id'])
    usr_year2distances_df_external = dict_to_df(usr_year2distances, cols, index_name=['user_id', year_colname])
    usr_quarter2distances_df_external = dict_to_df(usr_quarter2distances, cols, index_name=['user_id', quarter_colname])
    
    usr_df = usr2distances_df.join(usr2distances_df_external, lsuffix='_internal', rsuffix='_external', how='outer')
    usr_year_df = usr_year2distances_df.join(usr_year2distances_df_external, lsuffix='_internal', rsuffix='_external', how='outer')
    usr_quarter_df = usr_quarter2distances_df.join(usr_quarter2distances_df_external, lsuffix='_internal', rsuffix='_external', how='outer')
    return (usr_df, usr_year_df, usr_quarter_df)

def i_we_pca(df, col):
    new_df = df.dropna(axis=0, subset=[col])
    matrix = np.array(np.array(new_df[col]).tolist())
    pca = PCA(n_components=1)
    new_df.insert(2, col+'_one', list(pca.fit_transform(matrix)))
    pca = PCA(n_components=5)
    new_df.insert(2, col+'_five', list(pca.fit_transform(matrix)))
    pca = PCA(n_components=10)
    new_df.insert(2, col+'_ten', list(pca.fit_transform(matrix)))
    return df.join(new_df[[col+'_one', col+'_five', col+'_ten']])

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
    
    usr2measures, usr2annual_measures, usr2quarterly_measures = reading_embeddings(embeddings_dir, test)
    for col in ['i_we_diff_internal', 'i_we_cluster_diff_internal', 'i_we_symmetric_diff_internal']:
        usr2quarterly_measures = i_we_pca(usr2quarterly_measures, col)
    
    # different embedding files should be matched with the same hr file as hr data is not in panel format
    sys.stderr.write('Reading HR and Survey data at %s.\n' % datetime.now())
    hr_df = extract_hr_survey_df(survey_filename, user_qualtrics_file, users_file, perf_likert, perf_percentage)

    # could just merge and write to csv without calling another function
    sys.stderr.write('Outputting dataframe at %s.\n' % datetime.now())
    if usr2measures is not None: hr_df.join(usr2measures).to_csv(user_output_filename)
    if usr2annual_measures is not None: hr_df.join(usr2annual_measures).to_csv(annual_output_filename)
    if usr2quarterly_measures is not None: hr_df.join(usr2quarterly_measures).to_csv(quarterly_output_filename)
    
    sys.stderr.write("Finished outputting measures at %s, with a duration of %s.\n"
        % (str(datetime.now()), str(datetime.now() - starttime)))


