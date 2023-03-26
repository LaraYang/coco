#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    3b_output_alternative_measuress.py test|actual
If a Negative Dimension Error occurs, check to see if there are empty embeddings
Two types of results are currently computed
1) Projections
2) Raw count frequencies
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

user_output_filename = os.path.join(output_dir, "embeddings_users_{}d_alternative_mincount{}.csv".format(embedding_dim, mincount))
annual_output_filename = os.path.join(output_dir, "embeddings_annual_{}d_alternative_mincount{}csv".format(embedding_dim, mincount))
quarterly_output_filename = os.path.join(output_dir, "embeddings_quarterly_{}d_alternative_mincount{}.csv".format(embedding_dim, mincount))

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

file_name_re = re.compile("5([a-z0-9]+)_(2020(Q3)?_)?internal.txt")
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
    relevant_dims = ["we_dim", "we_they_dim", "family_dim", "belonging_dim", "pride_dim", "valence_dim", "disciplined_dim", "competitive_dim", "passionate_dim", "responsive_dim"]
    num_cols = 2 + len(relevant_dims)*2
    dim_i_word2proj = {}
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
                tok2count = Counter(email['hb'].replace('\n', ' ').replace("SENT_END", "").strip().split())
                i_counts = sum([tok2count[h] for h in hash_pronouns[i_index:we_index]])
                we_counts = sum([tok2count[h] for h in hash_pronouns[we_index:]])
                projs = []
                for d in relevant_dims:
                    for i in range(2):
                        # calculating mean projection score per email
                        dim_projs = []
                        for t in toks:
                            if t in company_model.vocab:
                                key = (d, i, t)
                                if not key in dim_i_word2proj.keys():
                                    proj = project(company_model[t], name2dims[d][i])
                                    dim_i_word2proj[key] = proj
                                dim_projs.append(dim_i_word2proj[key])
                        projs.append(mean(dim_projs) if len(dim_projs) > 0 else 0)
                row = [i_counts, we_counts] + projs
                for col in range(num_cols):
                    usr2counts[user][col] += row[col]
                    usr_year2counts[(user, year)][col] += row[col]
                    usr_quarter2counts[(user, quarter)][col] += row[col]
        
        general_dims = ['i_we', 'we_they', 'family', 'belonging', "pride", "valence", "disciplined", "competitive", "passionate", "responsive"] 
    post = ['_mean_proj_toks', '_pca_proj_toks']
    cols = ['num_i_words', 'num_we_words'] + [d+p for d in general_dims for p in post]

    usr2counts_df = dict_to_df(usr2counts, cols, index_name=['user_id'])
    usr_year2counts_df = dict_to_df(usr_year2counts, cols, index_name=['user_id', year_colname])
    usr_quarter2counts_df = dict_to_df(usr_quarter2counts, cols, index_name=['user_id', quarter_colname])
    return (usr2counts_df, usr_year2counts_df, usr_quarter2counts_df)

def project_frequency_out(vectors, model):
    """
    Return cosine similarity between I-words and we-words after frequency dimension (assumed to be the top PCA dimension) is projected out of vectors.
    Currently unused in the script as it was shown to not be related to identification in earlier iterations of the code.
    Approach inspired by both Mu (2018) and Wang et al. (2020).
    """
    miu = np.mean(vectors, axis=0)
    demeaned_vectors = vectors - miu
    pca = PCA(n_components = embedding_dim)
    pca.fit(demeaned_vectors)
    frequency_dim = pca.components_[0]
    i = drop(model[hash_single_pronouns[0]]-miu, frequency_dim) if hash_single_pronouns[0] in model.vocab else None
    we = drop(model[hash_single_pronouns[1]]-miu, frequency_dim) if hash_single_pronouns[1] in model.vocab else None
    i_cluster = [drop(model[word]-miu, frequency_dim) for word in hash_pronouns[i_index:we_index] if word in model.vocab]
    i_cluster = None if len(i_cluster) == 0 else np.mean(i_cluster, axis=0)
    we_cluster = [drop(model[word]-miu, frequency_dim) for word in hash_pronouns[we_index:] if word in model.vocab]
    we_cluster = None if len(we_cluster) == 0 else np.mean(we_cluster, axis=0)
    embeds = ([cossim_with_none(i, we, vec_format='dense'), cossim_with_none(i_cluster, we_cluster, vec_format='dense')])
    return embeds

def embeddings_projections(model, name2dims):
    """
    Returns projection scores on different dimensions in dims. If interested in computing cosine similarities between i-words
    and we-words after de-meaning and projecting out top PCA component, call project_frequency_out.
    Parameters
    ----------
    model : gensim.models.Word2Vec
        Model that stores the embeddings for each word
    name2dims : dict
        Maps dimension names to 2-tuples of dimensions, where the first element is a dimension constructed through means of differences
        and the second dimension is a dimension constructed using top component in difference vectors
    Returns
    -------
    results : list
        A list of scalar projections
    """
    # rows are words and columns are features
    vectors = np.asarray(model.vectors)
    projs = [mean([project(v, name2dims[d][i]) for v in vectors]) for d in ['we_dim', 'we_they_dim', 'family_dim', 'belonging_dim', 'pride_dim', 'valence_dim'] for i in range(2)]
    we_hash, i_hash, org_hash = hash_pronouns[we_index], hash_pronouns[i_index], word2hash['collabera']

    projs.extend([project(model[org_hash], name2dims[d][i]) if org_hash in model.vocab else np.nan for d in ['we_dim', 'we_they_dim', 'family_dim', 'belonging_dim', 'pride_dim', 'valence_dim'] for i in range(2)])
    projs.extend([project(model[we_hash], name2dims[d][i]) if we_hash in model.vocab else np.nan for d in ['family_dim', 'belonging_dim', 'pride_dim', 'valence_dim'] for i in range(2)])
    projs.extend([mean([project(v, name2dims[d][i]) for v in vectors]) for d in ['disciplined_dim', 'competitive_dim', 'passionate_dim', 'responsive_dim'] for i in range(2)])
    projs.extend([project(model[i_hash], name2dims[d][i]) if i_hash in model.vocab else np.nan for d in ['disciplined_dim', 'competitive_dim', 'passionate_dim', 'responsive_dim'] for i in range(2)])
    projs.extend([abs(project(model[we_hash], name2dims[d][i]) - project(model[i_hash], name2dims[d][i])) if we_hash in model.vocab and i_hash in model.vocab else np.nan for d in ['disciplined_dim', 'competitive_dim', 'passionate_dim', 'responsive_dim'] for i in range(2)])
    projs.append(word_similarity(model, hash_pronouns[i_index:we_index], org_hash))
    return projs

def process_single_embedding_file(i, num_files, embeddings_dir, file, name2dims):
    """
    Reading from one embedding file to generate projection scores and embedding similarities.
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
    name2dims : dict
        Dictionary mapping dimension names to dimensions to project on
    Returns
    -------
    list of doubles
        A list of projection scores - scalar projections on all dim in name2dims
    """
    mittens_file = os.path.join(embeddings_dir, file)
    sys.stderr.write("Processing \t%d/%d -'%s', at %s.\n" % (i, num_files, mittens_file, datetime.now()))
    # chopping off the file extension in filename
    tmp_mittens = os.path.join(tmp_dir, file[0:-4] + "_word2vec.txt")
    try:
        word2vec_mittens_file = get_tmpfile(tmp_mittens)
        glove2word2vec(mittens_file, word2vec_mittens_file)
        model = KeyedVectors.load_word2vec_format(word2vec_mittens_file)
        return embeddings_projections(model, name2dims)
    except Exception as e:
        sys.stderr.write('File %s caused an error: %s.\n' % (mittens_file, str(e)))
        print(traceback.format_exc())

def reading_embeddings(embeddings_dir, name2dims, test_mode):
    """
    Workhouse function for processing each individual embedding file to calculate projection scores and to calculate
    cos(i, we) after projecting out top component.
    Parameters
    ----------
    embeddings_dir : str
        Directory where all embeddings live
    name2dims : dict
        A dictionary matching dimension names to dimensions represented by numpy vectors
    test_mode : bool
        If true, restrict number of files
    Returns
    -------
    tuple
        A 3-tuple of pd.DataFrames
    """
    general_dims = ['i_we', 'we_they', 'family', 'belonging', "pride", "valence"]
    value_dims = ["disciplined", "competitive", "passionate", "responsive"]
    post = ['_mean_proj', '_pca_proj']
    cols = ([name+p for name in general_dims for p in post] +
            [name+p+'_org' for name in general_dims for p in post] +
            [name+p+'_we' for name in general_dims[2:] for p in post] + 
            [name+p for name in value_dims for p in post] +
            [name+p+'_i' for name in value_dims for p in post] + 
            [name+p+'_i_we' for name in value_dims for p in post] +
            ['i_org'])
    
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
        sys.stderr.write('Iteration %d: Calculate alternative measures for %d files at %s.\n' % (i, num_files, str(datetime.now())))
        usr2distances, usr_year2distances, usr_quarter2distances = defaultdict(list), defaultdict(list), defaultdict(list)
        pool = multiprocessing.Pool(processes = num_cores)
        results = {}
        for i, file in enumerate(files, 1):
            usr, time_key = extract_variables_from_file(file)
            results[(usr, time_key)] = pool.apply_async(process_single_embedding_file, args=(i, num_files, embeddings_dir, file, name2dims))
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
        
        result_dfs.append([dict_to_df(usr2distances, cols, index_name=['user_id']),
            dict_to_df(usr_year2distances, cols, index_name=['user_id', year_colname]),
            dict_to_df(usr_quarter2distances, cols, index_name=['user_id', quarter_colname])])

    usr_df = result_dfs[0][0].join(result_dfs[1][0], lsuffix='_internal', rsuffix='_external', how='outer')
    usr_year_df = result_dfs[0][1].join(result_dfs[1][1], lsuffix='_internal', rsuffix='_external', how='outer')
    usr_quarter_df = result_dfs[0][2].join(result_dfs[1][2], lsuffix='_internal', rsuffix='_external', how='outer')
    return (usr_df, usr_year_df, usr_quarter_df)   

def build_all_dimensions():
    """
    Returns a dictionary that matches dimension name to a 2-tuple of dimensions, where each dimension is represented using a numpy vector.
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
    tmp_mittens = os.path.join(tmp_dir, "mittens_embeddings_all_word2vec.txt")
    word2vec_mittens_file = get_tmpfile(tmp_mittens)
    glove2word2vec(company_embeddings_filename, word2vec_mittens_file)
    company_model = KeyedVectors.load_word2vec_format(word2vec_mittens_file)

    sys.stderr.write("Building all dimensions at %s.\n" % datetime.now())
    name2dims = build_all_dimensions()

    sys.stderr.write('Loading corpus counts at %s.\n' % datetime.now())
    usr2counts, usr2annual_counts, usr2quarterly_counts = read_raw_email(activity_file, email_file, name2dims, test)

    sys.stderr.write('Reading embeddings at %s.\n' % datetime.now())
    usr2measures, usr2annual_measures, usr2quarterly_measures = reading_embeddings(embeddings_dir, name2dims, test)
    
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
