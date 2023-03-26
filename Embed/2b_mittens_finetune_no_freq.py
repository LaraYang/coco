#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    2b_mittens_finetune_no_freq.py actual|test
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
import csv
from utils import *
import random
from sklearn.decomposition import PCA

num_cores = 10
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
mincount = 150
max_iter = 1000

home_dir = "/ifs/projects/amirgo-identification/"
email_dir = os.path.join(home_dir, "email_data/")
mittens_dir = os.path.join(home_dir, "mittens")
utils_dir = os.path.join(mittens_dir, "utils")
output_dir = os.path.join(mittens_dir, "embeddings_{}d_mincount{}_no_freq".format(embedding_dim, mincount))
test_dir = os.path.join(mittens_dir, "embeddings_test_{}d_mincount{}_no_freq".format(embedding_dim, mincount))
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
target_file = 'target_users.txt'

collabera_hashes = set([v for k, v in domain_hash.items()])

def load_target_users(target_file, regenerate=False):
    """
    Returns a set of User IDs, as used in anonymized emails, for whom survey-based identification data is available
    Parameters
    ----------
    target_file : str
        Full filepath to target users
    regenerate : bool, optional
        If true, regenerate target user file even if it already exists
    """
    if not regenerate and os.path.exists(target_file):
        with open(target_file, "r") as file:
            userids = []
            for line in file:
                userids.append(line.strip()) # removing newline
            return set(userids)
    else:
        survey_df = pd.read_csv(survey_filename)
        uids_responded = survey_df.dropna(subset=['bergami_org_num'])['uid'].to_list()
        uids_responded += survey_df.dropna(subset=['mael_avg'])['uid'].to_list()
        uids_responded = list(set(uids_responded))
        user_qualtrics_df = pd.read_csv(user_qualtrics_file)
        user_qualtrics_df = user_qualtrics_df[user_qualtrics_df['UID'].isin(uids_responded)]
        emails_responded = user_qualtrics_df['Email']
        email2uid = {}
        with open(users_file, encoding='utf-8') as f:
            for line in f:
                user = json.loads(line)
                for e in user['Emails']:
                    email2uid[e] = user['UserId']
        userids = []
        for e in emails_responded:
            if e in email2uid.keys():
                userids.append(email2uid[e])
            else:
                print(e)
        with open(target_file, "w") as file:
            for u in userids:
                file.write(u+'\n')
        return set(userids)

def load_user_emails(target_users=None, test_mode=False):
    """
    Read emails and activities to return a dictionary that matches users to all their emails
    Parameters
    ----------
    target_users : list
        If provided, only emails of individuals with user IDs included in this set are used for computation
        If None, all users are included for computation
    test_mode : bool, optional
        If testing, only load partial emails
    Returns
    -------
    uid2emails : dict
        A dictionary matching user ids to a list of emails of type dict
    """
    uid2emails = defaultdict(list)
    sid2activity = {}
    with open(activity_file, encoding='utf-8') as f:
        for line in f:
            activity = json.loads(line)
            if target_users is None or activity['UserId'] in target_users:
                sid2activity[activity['MailSummarySid']] = activity
    target_sids = sid2activity.keys()
    with open(email_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if test_mode and i > 100000:
                break
            email = json.loads(line)
            if email['sid'] in target_sids:
                lang = email['l']
                if len(email['hb']) > 0 and lang[0] == "__label__en" and lang[1] > 0.5:
                    # check internal or external or mixed
                    activity = sid2activity[email['sid']]
                    recipients = get_recipients(activity)
                    pure_internal, pure_external = True, True
                    for r in recipients:
                        domain = r.split('@')[1]
                        if pure_external and domain in collabera_hashes:
                            pure_external = False
                        elif pure_internal and domain not in collabera_hashes:
                            pure_internal = False
                    if (pure_external == True and pure_internal == True):
                        assert not (pure_external == True and pure_internal == True) 
                    if pure_external:
                        activity['email_type'] = 'ext'
                    elif pure_internal:
                        activity['email_type'] = 'int'
                    else:
                        activity['email_type'] = 'mixed'
                    # combine both JSONs into one
                    activity['hb'] = email['hb']
                    uid2emails[activity['UserId']].append(activity)
    return uid2emails

def process_user(i, num_users, uid, emails, email_type, timekeys=None):
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
    emails : list of dict
        A list of emails converted from JSON formats that include both content and meta-data
    email_type : str
        Specifies which types of emails to include when building embeddings
    timekeys : list or None, optional
        If not None, only training embeddings for specified time periods
    """
    sys.stderr.write("\nProcessing \t%d/%d - user '%s' %s emails, at %s.\n" % (i, num_users, uid, email_type, datetime.now()))
    user_embedding_filename = os.path.join(output_dir, "{}_{}.txt".format(uid, email_type))
    if not os.path.exists(user_embedding_filename):
        X = build_weighted_matrix(emails, mincount=mincount, window_size=window_size, email_type=email_type)
        if X.empty:
            return
        mittens = Mittens(n=embedding_dim, max_iter=max_iter, mittens=mittens_params)
        mittens = mittens.fit(
            X.values, 
            vocab=list(X.index), 
            initial_embedding_dict=company_embeddings)
        mittens_df = pd.DataFrame(mittens, index=X.index)
        if not mittens_df.empty:
            output_embeddings(mittens_df, filename=user_embedding_filename)
    user_embeddings = glove2dict(user_embedding_filename)
    sliced_usr_corpus = slice_user_corpus(emails, "all")
    if timekeys is None:
        timekeys = sliced_usr_corpus.keys()
    for timekey in timekeys:
        emails = sliced_usr_corpus[timekey]
        user_embedding_time_filename = os.path.join(output_dir, "{}_{}_{}.txt".format(uid, timekey, email_type))
        if not os.path.exists(user_embedding_time_filename):            
            X = build_weighted_matrix(emails, mincount=mincount, window_size=window_size, email_type=email_type)
            if X.empty:
                continue
            mittens = Mittens(n=embedding_dim, max_iter=max_iter, mittens=mittens_params)
            if not user_embeddings:
                sys.stderr.write("\n%s does not have corresponding user embeddings with timekey %s.\n" % (usr, timekey))
            mittens = mittens.fit(
                X.values,
                vocab=list(X.index), 
                initial_embedding_dict=user_embeddings)
            mittens_df = pd.DataFrame(mittens, index=X.index)
            if not mittens_df.empty:
                output_embeddings(mittens_df, filename=user_embedding_time_filename)
    return

def process_users(uid2emails, test_mode, email_types=['internal'], timekeys=None):
    """
    Processes emails of each user in parallel
    Parameter
    ---------
    uid2emails : dict
        A dictionary matching user IDs to a list of emails converted from JSON to dictionaries
    test_mode : bool
        If in test_mode, only process num_users_to_test users
    email_types : list, optional
        A list of email_types to process, either internal, external, or both
    timekeys : list, optional
        Timekeys that indicate time-specific embeddings to be computed
    """
    num_users = len(uid2emails)
    if test_mode:
        keys = random.sample(list(uid2emails), num_users_to_test)
        uid2emails = {k : uid2emails[k] for k in keys}
        num_users = num_users_to_test
    sys.stderr.write('Processing %d users in parallel at %s.\n' % (num_users, str(datetime.now())))

    pool = multiprocessing.Pool(processes = num_cores)
    results = [pool.apply_async(process_user, args=(i, num_users, uid, uid2emails[uid], email_type, timekeys, )) for i, uid in enumerate(uid2emails) for email_type in email_types]
    for r in results:
        r.get()
    pool.close()
    pool.join()
    return

if __name__ == '__main__':
    starttime = datetime.now()
    test_mode = sys.argv[1].lower() == 'test' or sys.argv[1].lower() == 't'
    for d in [mittens_dir, utils_dir, output_dir, test_dir]:
        if not os.path.exists(d):
            os.mkdir(d)
    if test_mode:
        output_dir = test_dir

    sys.stderr.write("Loading target_users at {}.\n".format(str(datetime.now())))
    target_users = sorted(list(load_target_users(target_file, regenerate=True)))
    
    sys.stderr.write("Loading all emails and activities for {} users at {}.\n".format(str(len(target_users)), str(datetime.now())))
    company_embeddings = remove_frequency(glove2dict(company_embeddings_filename), embedding_dim)
    
    third = len(target_users) // 3
    user_chunks = [target_users[0:third], target_users[third:third*2], target_users[third*2:]]
    for i, users in enumerate(user_chunks):
        uid2emails = load_user_emails(users, test_mode)
        if test_mode:
            sys.stderr.write("Processing test users in chunk {} at {}.\n".format(i, str(datetime.now())))
        else:
            sys.stderr.write("Processing all users in chunk {} at {}.\n".format(i, str(datetime.now())))
        # Remove these arguments to compute all emails
        process_users(uid2emails, test_mode, email_types=['internal', 'external'])

    sys.stderr.write("\n\nFinished processing at {}, with a duration of {}\n".format(str(datetime.now()), str(datetime.now()-starttime)))