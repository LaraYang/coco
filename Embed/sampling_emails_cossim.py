#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    sampling_emails_cossim.py
Samples over individuals and emails to determine the mechanical relationship between cosine similarity and number of emails
"""
import re
import os
import ujson as json
import sys
import multiprocessing
import shutil
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
from mittens import Mittens
from utils import *
from numpy import random
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.matutils import cossim, any2sparse

# trying yens10
num_cores = 44
num_users_to_test = 60
""" mittens_params: float (default: 0.1)
        Relative weight assigned to remaining close to the original
        embeddings. Setting to 0 means that representations are
        initialized with the original embeddings but there is no
        penalty for deviating from them. Large positive values will
        heavily penalize any deviation, so the statistics of the new
        co-occurrence matrix will be increasingly ignored. A value of
        1 is 'balanced': equal weight is given to the GloVe objective
        on the new co-occurrence matrix and to remaining close to the
        original embeddings. 0.1 (the default) is recommended. See [2]_.
"""
# default
mittens_params = 0.1
# moving this larger moves out of the default window we have seen in papers (1-10)
window_size = 10
# seems like smallest embedding dim works best given that there might not that many dimensions needed to capture the difference between i and we
embedding_dim = 50
# moving this upper would provide higher quality embeddings but lead to smaller sample size
# 300 is mitten's default, 150 provides more samples and thus is the default for this project (when no mincount is specified in output fields, mincount=150)
mincount = 150
max_iter = 1000

home_dir = "/ifs/projects/amirgo-identification/"
email_dir = os.path.join(home_dir, "email_data/")
mittens_dir = os.path.join(home_dir, "mittens")
utils_dir = os.path.join(mittens_dir, "utils")
tmp_dir = os.path.join(mittens_dir, "tmp")
output_dir = os.path.join(mittens_dir, "embeddings_resample_{}d_mincount{}".format(embedding_dim, mincount))
output_file = os.path.join(home_dir, "email_idtf_data", "embeddings_resample_{}d_mincount{}.csv".format(embedding_dim, mincount))
test_file = os.path.join(home_dir, "email_idtf_data", "test_embeddings_resample_{}d_mincount{}.csv".format(embedding_dim, mincount))

email_file = os.path.join(email_dir, 'MessagesHashed.jsonl')
users_file = os.path.join(email_dir, 'Users.json')
activity_file = os.path.join(email_dir, 'Activities.json')
user_qualtrics_file = os.path.join(home_dir, "survey_hr_data", "UsersQualtrics.csv")
company_embeddings_filename = "/ifs/gsb/amirgo/spacespace/spacespace/Coco/Embed/GloVe-master/vectors_{}d.txt".format(embedding_dim)
survey_filename = "/ifs/gsb/amirgo/spacespace/spacespace/Coco/analyses_data/preprocessed_survey_hr.csv"
domain_hash = {
    'collabera.com':                     '509c8f6b1127bceefd418c023533d653', 
    'collaberainc.mail.onmicrosoft.com': 'ec5b67548b6ec06f1234d198efec741e', 
    'collaberainc.onmicrosoft.com':      '86160680578ee9258f097a67a5f25af9', 
    'collaberasp.com':                   '6bf3934d19f1acf5b9295b63e0e7f66e', 
    'g-c-i.com':                         '3444d1f7d5e46443080f2d069e41a10c'}

collabera_hashes = set([v for k, v in domain_hash.items()])

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

def load_user_emails(num_users, sample_size, num_samples, repeat_sampling, test_mode=False):
    """
    The main workhorse function for obtaining repeatedly sampled emails for the same target user
    Parameters
    ----------
    num_users : int
        Specifies how many users we want to resample
    sample_size : int
        Specifies how large a single sample should be
    num_samples : list of int
        Specifies how many times should samples of sample size be resampled
    repeat_sampling : bool
        If repeat_sampling, the same emails are repeated num_samples times. Else, different emails
        are drawn num_samples times.
    Returns
    -------
    uid2size2emails
        A dictionary mapping users to a dictionary mapping different sizes to resampled emails
    """
    uids = list()
    sid2activity = dict()
    with open(activity_file, encoding='utf-8') as f:
        for line in f:
            activity = json.loads(line)
            uids.append(activity['UserId'])
            sid2activity[activity['MailSummarySid']] = activity

    target_uids = random.choice(uids, num_users)
    target_uids2emails = defaultdict(list)
    with open(email_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if test_mode and i > 100000:
                break
            email = json.loads(line)
            user_id = sid2activity[email['sid']]['UserId']
            if user_id in target_uids:
                lang = email['l']
                if len(email['hb']) > 0 and lang[0] == "__label__en" and lang[1] > 0.5:
                    # no need restricting to internal emails as we are just testing a mechanical relationship
                    target_uids2emails[user_id].append(email)
    target_uids2indices2values = defaultdict(dict)
    for uid, emails in target_uids2emails.items():
        target_uids2indices2values[uid]['num_emails'] = len(emails)
        if repeat_sampling:
            target_emails = list(random.choice(emails, sample_size))
            for n in num_samples:
                target_uids2indices2values[uid][n] = target_emails*n
        else:
            for n in num_samples:
                tot_size = sample_size*n
                if tot_size <= len(emails):
                    target_uids2indices2values[uid][n] = list(random.choice(emails, tot_size, replace=False))
    return target_uids2indices2values

def process_user(i, num_users, uid, samples, num_samples):
    """
    Workhorse function for training embedding spaces for each individual.
    Parameters
    ----------
    i : int
        Index of current user used to keep track of progress
    num_users : int
        Total number of users used to keep track of progress
    uid : str
        User ID of current user
    samples : dict of lists
        A dictionary mapping different sample sizes to lists of emails that have been resampled
    num_samples : list of int
        A list that specifies the index of samples to be iterated through
    Returns
    -------
    dict
        A dictionary mapping user id to a dictionary that maps number of repeat samples and cosine similarity 
    """
    sys.stderr.write("\nProcessing \t%d/%d - user '%s' emails, at %s.\n" % (i, num_users, uid, datetime.now()))
    sample2cossim = {}
    for n in num_samples:
        if n in samples.keys():
            emails = samples[n]
            user_embedding_filename = os.path.join(output_dir, "{}_sample_{}.txt".format(uid, n))
            if not os.path.exists(user_embedding_filename):
                X = build_weighted_matrix(emails, mincount=mincount, window_size=window_size, email_type='all')
                if X.empty:
                    sample2cossim[n] = np.nan
                    continue
                mittens = Mittens(n=embedding_dim, max_iter=max_iter, mittens=mittens_params)
                mittens = mittens.fit(
                    X.values, 
                    vocab=list(X.index), 
                    initial_embedding_dict=company_embeddings)
                mittens_df = pd.DataFrame(mittens, index=X.index)
                if not mittens_df.empty:
                    output_embeddings(mittens_df, filename=user_embedding_filename)
            tmp_mittens = os.path.join(tmp_dir, "{}_sample_{}_word2vec.txt".format(uid, n))
            word2vec_mittens_file = get_tmpfile(tmp_mittens)
            glove2word2vec(user_embedding_filename, word2vec_mittens_file)
            model = KeyedVectors.load_word2vec_format(word2vec_mittens_file)
            sample2cossim[n] = word_similarity(model, hash_pronouns[i_index:we_index], hash_pronouns[we_index:])
        # if n not in sample, it means the user did not have sufficient emails to have sample n*sample_size emails sampled from 
        # all its emails
        else:
            sample2cossim[n] = np.nan
    sample2cossim['num_emails'] = samples['num_emails']
    return {uid : sample2cossim}
        
def process_users(uid2samples, num_samples):
    """
    Processes emails of each user in parallel
    Parameter
    ---------
    uid2samples: dict of dict of emails
        Maps user ids to different samples
    """
    num_users = len(uid2samples)
    sys.stderr.write('Processing %d users in parallel at %s.\n' % (num_users, str(datetime.now())))
    pool = multiprocessing.Pool(processes = num_cores)
    results = [pool.apply_async(process_user, args=(i, num_users, uid, uid2samples[uid], num_samples, )) for i, uid in enumerate(uid2samples)]
    uid2sample2cossim = dict()
    for r in results:
        uid2sample2cossim.update(r.get())
    pool.close()
    pool.join()
    return uid2sample2cossim

if __name__ == '__main__':
    starttime = datetime.now()
    for d in [mittens_dir, utils_dir, output_dir]:
        if not os.path.exists(d):
            os.mkdir(d)
    
    num_users = 200
    sample_size = 100
    num_samples = [5, 25, 50, 75, 100]
    test_mode = sys.argv[1].lower() == 'test'
    if test_mode:
        num_users = 5
        output_file = test_file
    
    sys.stderr.write("Loading user emails at %s.\n" % str(datetime.now()))
    # output that contains repeat_sample corresponds to repeat_sampling = True
    # output that contains resample corresponds to repeat_sampling = False
    repeat_sampling = False
    sampled = load_user_emails(num_users, sample_size, num_samples, repeat_sampling, test_mode)

    company_embeddings = glove2dict(company_embeddings_filename)
    sys.stderr.write("Processing emails at %s.\n" % str(datetime.now()))
    uid2sample2cossim = process_users(sampled, num_samples)
    
    keys = ['num_emails'] + num_samples
    cols = ['num_emails'] + ['sample_' + str(n) for n in num_samples]
    uid2rows = {user_id : [samples[k] for k in keys] for user_id, samples in uid2sample2cossim.items()}
    df = dict_to_df(uid2rows, cols, index_name=['user_id'])
    df.to_csv(output_file)
    sys.stderr.write("Finished outputting repeat sample measures at %s, with a duration of %s.\n"
        % (str(datetime.now()), str(datetime.now() - starttime)))
