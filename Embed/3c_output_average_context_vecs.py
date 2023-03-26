#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    3c_output_average_context_vecs.py test|actual
If a Negative Dimension Error occurs, check to see if there are empty embeddings
Uses context vectors to build i, we, and orgname embeddings
Calculate cosine similarity between i and we
Project orgname on different dimensions
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

home_dir = "/ifs/projects/amirgo-identification/"
email_dir = os.path.join(home_dir, "email_data/")
mittens_dir = os.path.join(home_dir, "mittens")
glove_dir = "/ifs/gsb/amirgo/spacespace/spacespace/Coco/Embed/GloVe-master/"
utils_dir = os.path.join(mittens_dir, "utils")
email_file = os.path.join(email_dir, 'MessagesHashed.jsonl')
users_file = os.path.join(email_dir, 'Users.json')
activity_file = os.path.join(email_dir, 'Activities.json')
survey_dir = os.path.join(home_dir, "survey_hr_data")
user_qualtrics_file = os.path.join(survey_dir, "UsersQualtrics.csv")
perf_percentage = os.path.join(survey_dir, "perf_rating_percentages.csv")
perf_likert = os.path.join(survey_dir, "perf_rating_likert.csv")

embedding_dim = 300
analyses_data_dir = "/ifs/gsb/amirgo/spacespace/spacespace/Coco/analyses_data/"
survey_filename = os.path.join(analyses_data_dir, "preprocessed_survey_hr.csv")

company_embeddings_filename = os.path.join(glove_dir, "vectors_{}d.txt".format(embedding_dim))
new_company_embeddings_filename = os.path.join(glove_dir, "vectors_no_freq_{}d.txt".format(embedding_dim))

tmp_dir = os.path.join(mittens_dir, "tmp")
output_dir = os.path.join(home_dir, "email_idtf_data")
user_output_filename = os.path.join(output_dir, "embeddings_users_avg_context_{}d.csv".format(embedding_dim))
annual_output_filename = os.path.join(output_dir, "embeddings_annual_avg_context_{}d.csv".format(embedding_dim))
quarterly_output_filename = os.path.join(output_dir, "embeddings_quarterly_avg_context_{}d.csv".format(embedding_dim))


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
hash_collabera = '974811d0'
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
    test_mode: bool
        If true, run a small number of files
    Returns
    -------
    tuple
        A tuple of user-level, annual, and quarterly dataframes
    """
    sid2activity = {}
    cols = ['num_tokens', 'num_messages', 'i_context_fixed', 'we_context_fixed', 'org_context_fixed', 'i_context_sent', 'we_context_sent', 'org_context_sent',
    'i_context_fixed_num', 'we_context_fixed_num', 'org_context_fixed_num', 'i_context_sent_num', 'we_context_sent_num', 'org_context_sent_num']
    num_cols = len(cols)
    window_size = 10
    usr2counts, usr_year2counts, usr_quarter2counts = (
        defaultdict(lambda : [0, 0, np.zeros(embedding_dim), np.zeros(embedding_dim), np.zeros(embedding_dim), np.zeros(embedding_dim), np.zeros(embedding_dim), np.zeros(embedding_dim), 0, 0, 0, 0, 0, 0]),
        defaultdict(lambda : [0, 0, np.zeros(embedding_dim), np.zeros(embedding_dim), np.zeros(embedding_dim), np.zeros(embedding_dim), np.zeros(embedding_dim), np.zeros(embedding_dim), 0, 0, 0, 0, 0, 0]),
        defaultdict(lambda : [0, 0, np.zeros(embedding_dim), np.zeros(embedding_dim), np.zeros(embedding_dim), np.zeros(embedding_dim), np.zeros(embedding_dim), np.zeros(embedding_dim), 0, 0, 0, 0, 0, 0]))
    
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
                toks = email['hb'].replace('\n', ' ').replace("SENT_END", "").strip().split()
                num_toks = len(toks)
                i_indices = [i for i, tok in enumerate(toks) if tok in hash_pronouns[i_index:we_index]]
                we_indices = [i for i, tok in enumerate(toks) if tok in hash_pronouns[we_index:]]
                org_indices = [i for i, tok in enumerate(toks) if tok == hash_collabera]
                i_context_fixed, we_context_fixed, org_context_fixed, i_context_sent, we_context_sent, org_context_sent = np.zeros(embedding_dim), np.zeros(embedding_dim), np.zeros(embedding_dim), np.zeros(embedding_dim), np.zeros(embedding_dim), np.zeros(embedding_dim)
                i_context_fixed_num, we_context_fixed_num, org_context_fixed_num, i_context_sent_num, we_context_sent_num, org_context_sent_num = 0, 0, 0, 0, 0, 0
                for i in i_indices:
                    # we want to exclude the number in the middle
                    for j in range(i-window_size, i+window_size+1):
                        if j >= 0 and j < len(toks) and j != i and j not in i_indices and j not in we_indices and toks[j] in company_model.vocab:
                            i_context_fixed += company_model[toks[j]]
                            i_context_fixed_num += 1
                for i in we_indices:
                    for j in range(i-window_size, i+window_size+1):
                        if j >= 0 and j < len(toks) and j != i and j not in i_indices and j not in we_indices and toks[j] in company_model.vocab:
                            we_context_fixed += company_model[toks[j]]
                            we_context_fixed_num += 1
                for i in org_indices:
                    for j in range(i-window_size, i+window_size+1):
                        if j >= 0 and j < len(toks) and j != i and j not in org_indices and toks[j] in company_model.vocab:
                            org_context_fixed += company_model[toks[j]]
                            org_context_fixed_num += 1
                
                sents = [sent.strip() for sent in re.split('\n|SENT_END', email['hb']) if len(sent.strip()) > 0]
                for sent in sents:
                    i_indices, we_indices, org_indices = list(), list(), list()
                    toks = sent.split()
                    for i, tok in enumerate(toks):
                        if tok in hash_pronouns[i_index:we_index]:
                            i_indices.append(i)
                        elif tok in hash_pronouns[we_index:]:
                            we_indices.append(i)
                        elif tok == hash_collabera:
                            org_indices.append(i)
                    if len(i_indices) > 0 and len(we_indices) == 0:
                        for i, tok in enumerate(toks):
                            if i not in i_indices and tok in company_model.vocab:
                                i_context_sent += company_model[tok]
                                i_context_sent_num += 1
                    elif len(i_indices) == 0 and len(we_indices) > 0:
                        for i, tok in enumerate(toks):
                            if i not in we_indices and tok in company_model.vocab:
                                we_context_sent += company_model[tok]
                                we_context_sent_num += 1
                    elif len(org_indices) > 0:
                        for i, tok in enumerate(toks):
                            if tok in company_model.vocab:
                                org_context_sent += company_model[tok]
                                org_context_sent_num += 1

                    # excluding sentences where both i-words and we-words appear
                row = ([num_toks, 1, i_context_fixed, we_context_fixed, org_context_fixed, i_context_sent, we_context_sent, org_context_sent,
                    i_context_fixed_num, we_context_fixed_num, org_context_fixed_num, i_context_sent_num, we_context_sent_num, org_context_sent_num])
                for col in range(num_cols):
                    usr2counts[user][col] += row[col]
                    usr_year2counts[(user, year)][col] += row[col]
                    usr_quarter2counts[(user, quarter)][col] += row[col]
   
    usr2counts_df = dict_to_df(usr2counts, cols, index_name=['user_id'])
    usr_year2counts_df = dict_to_df(usr_year2counts, cols, index_name=['user_id', year_colname])
    usr_quarter2counts_df = dict_to_df(usr_quarter2counts, cols, index_name=['user_id', quarter_colname])

    for df in [usr2counts_df, usr_year2counts_df, usr_quarter2counts_df]:
        df['i_we_fixed'] = df.apply(lambda row : cossim_with_none(row['i_context_fixed']/row['i_context_fixed_num'], row['we_context_fixed']/row['we_context_fixed_num'], vec_format='dense'), axis=1)
        df['i_we_sent'] = df.apply(lambda row : cossim_with_none(row['i_context_sent']/row['i_context_sent_num'], row['we_context_sent']/row['we_context_sent_num'], vec_format='dense'), axis=1)
        df['org_fixed'] = df.apply(lambda row : row['org_context_fixed']/row['org_context_fixed_num'], axis=1)
        df['org_sent'] = df.apply(lambda row : row['org_context_sent']/row['org_context_sent_num'], axis=1)
        for name, dim in name2dims.items():
            dim_name = name.split('_')[0]
            df[dim_name+'_org_fixed'] = df.apply(lambda row: project(row['org_fixed'], dim[0]), axis=1)
            df[dim_name+'_org_sent'] = df.apply(lambda row : project(row['org_sent'], dim[0]), axis=1)
        
        df.drop(columns=['i_context_fixed', 'we_context_fixed', 'org_context_fixed', 'i_context_sent', 'we_context_sent', 'org_context_sent', 'org_fixed', 'org_sent'], inplace=True)
    return (usr2counts_df, usr_year2counts_df, usr_quarter2counts_df)

def build_all_dimensions():
    """
    Returns a dictionary that matches dimension name to a 2-tuple of dimensions, where eac dimension is represented using a numpy vector.
    """
    name2hashes = {'family_dim': ([word2hash[word] for word in ['family', 'home', 'community', 'team']],
        [word2hash[word] for word in ['money', 'pay', 'salary', 'income']]),
        'valence_dim': ([word2hash[word] for word in ["good", "great", "best"]],
            [word2hash[word] for word in ["bad", "awful", "worst"]]),
        'belonging_dim': ([word2hash[word] for word in ['included', 'attached']],
            [word2hash[word] for word in ['excluded', 'detached']]),
        'pride_dim': ([word2hash[word] for word in ["pride", "dignity", "honor"]],
            [word2hash[word] for word in ["shame", "humiliation", "dishonor"]]),
        'passionate_dim': ([word2hash[word] for word in ["passionate"]],
            [word2hash[word] for word in ["indifferent"]]),
        'competitive_dim': ([word2hash[word] for word in ["competitive"]], # noncompetitive is not included in GloVe, thus this word-pair is restricted to one word
            [word2hash[word] for word in ["lazy"]]),
        'responsive_dim': ([word2hash[word] for word in ["responsive"]],
            [word2hash[word] for word in ["unresponsive"]]),
        'disciplined_dim': ([word2hash[word] for word in ["disciplined"]],
            [word2hash[word] for word in ["undisciplined"]]),
        'we_dim': (hash_pronouns[we_index:], hash_pronouns[i_index:we_index]),
        'we_they_dim': (hash_pronouns[we_index:], [word2hash[word] for word in ['they', 'them', 'their', 'theirs', 'themselves']])}
    dims = {k : build_dimension([company_model[h] for h in hashes[0]], [company_model[h] for h in hashes[1]]) for k, hashes in name2hashes.items()}
    return dims

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

    sys.stderr.write("Building company model at %s.\n" % datetime.now())
    # Projecting frequency out and writing to disk
    # company_embeddings = remove_frequency(glove2dict(company_embeddings_filename), embedding_dim)
    # freq_removed_df = pd.DataFrame.from_dict(company_embeddings, orient='index')
    # output_embeddings(freq_removed_df, new_company_embeddings_filename)

    # just a temporary file; name does not matter
    tmp_mittens = os.path.join(tmp_dir, "mittens_embeddings_all_word2vec.txt")
    word2vec_mittens_file = get_tmpfile(tmp_mittens)
    # if not interested in frequency-removed version, change file name to company_embeddings_filename
    glove2word2vec(company_embeddings_filename, word2vec_mittens_file)
    company_model = KeyedVectors.load_word2vec_format(word2vec_mittens_file)

    sys.stderr.write("Building all dimensions at %s.\n" % datetime.now())
    name2dims = build_all_dimensions()
    sys.stderr.write('Loading corpus counts at %s.\n' % datetime.now())
    usr2measures, usr2annual_measures, usr2quarterly_measures = read_raw_email(activity_file, email_file, name2dims, test)

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
