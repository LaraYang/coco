"""
Utils file for CoCo
"""
import os
import sys
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
from mittens import Mittens
import csv
from operator import itemgetter
import ujson as json
import re
from gensim.matutils import cossim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from statistics import mean 
from sklearn.decomposition import PCA

#########################################################################
########### Helper Functions for Generating Mittens Embeddings ##########
#########################################################################
def _window_based_iterator(toks, window_size, weighting_function):
    for i, w in enumerate(toks):
        yield w, w, 1
        left = max([0, i-window_size])
        for x in range(left, i):
            yield w, toks[x],weighting_function(x)
        right = min([i+1+window_size, len(toks)])
        for x in range(i+1, right):
            yield w, toks[x], weighting_function(x)
    return

def glove2dict(glove_filename):
    """
    Reads word vectors into a dictionary
    Parameters
    ----------
    glove_filename : str
        Name of file that contains vectors
    Returns
    -------
    data : dict
        A dictionary matching words to their vectors
    """
    with open(glove_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        data = {line[0]: np.array(list(map(float, line[1: ]))) for line in reader}
    return data

# Inspired by original build_weighted_matrix in utils.py in the Mittens paper source codebase
def build_weighted_matrix(emails,
        mincount=300, vocab_size=None, window_size=10,
        weighting_function=lambda x: 1 / (x + 1),
        email_type='internal'):
    """
    Builds a count matrix based on a co-occurrence window of
    `window_size` elements before and `window_size` elements after the
    focal word, where the counts are weighted based on proximity to the
    focal word.
    Parameters
    ----------
    emails : list of dicts
        Emails converted from JSON formats
    mincount : int
        Only words with at least this many tokens will be included.
    vocab_size : int or None
        If this is an int above 0, then, the top `vocab_size` words
        by frequency are included in the matrix, and `mincount`
        is ignored.
    window_size : int
        Size of the window before and after. (So the total window size
        is 2 times this value, with the focal word at the center.)
    weighting_function : function from ints to floats
        How to weight counts based on distance. The default is 1/d
        where d is the distance in words.
    email_type : str, optional
        Specifies which types of emails to include when building embeddings
    Returns
    -------
    X : pd.DataFrame
        Cooccurence matrix guaranteed to be symmetric because of the way the counts are collected.
    """
    wc = defaultdict(int)
    corpus = read_corpus(emails, email_type=email_type, sentence_delim=False)
    if corpus is None:
        print("These emails are empty\t{}.\nEmpty corpus returned for email type {}".format(str(emails), email_type))
        return pd.DataFrame()
    for toks in corpus:
        for tok in toks:
            wc[tok] += 1
    if vocab_size:
        srt = sorted(wc.items(), key=itemgetter(1), reverse=True)
        vocab_set = {w for w, c in srt[: vocab_size]}
    else:
        vocab_set = {w for w, c in wc.items() if c >= mincount}
    vocab = sorted(vocab_set)
    n_words = len(vocab)
    # Weighted counts:
    counts = defaultdict(float)
    for toks in corpus:
        window_iter = _window_based_iterator(toks, window_size, weighting_function)
        for w, w_c, val in window_iter:
            if w in vocab_set and w_c in vocab_set:
                counts[(w, w_c)] += val
    X = np.zeros((n_words, n_words))
    for i, w1 in enumerate(vocab):
        for j, w2 in enumerate(vocab):
            X[i, j] = counts[(w1, w2)]
    X = pd.DataFrame(X, columns=vocab, index=pd.Index(vocab))
    return X

def read_corpus(emails, email_type, sentence_delim=False):
    """
    Parameters
    ----------
    emails : list of dict
        A list of emails converted from JSON formats
    email_type : str
        Specifies which types of emails to include when building embeddings
        'internal' filters for internal emails, 'external' filters for external and mixed emails, 
        and anything else does not filter
    sentence_delim : bool, optional
        If true, co-occurrences across sentence boundaries are ignored.
    Returns
    -------
    list of list of str
        Corpus converted from emails.
        If sentence_delim is false, returns a list of emails, which are represented as lists of tokens
        If sentence_delim is true, returns a list of sentences, which are represented as lists of tokens
    """
    # split with no argument splits on whitespaces and newlines
    if not sentence_delim:
        if email_type == 'internal':
            return [email['hb'].replace('\n', ' ').replace("SENT_END", "").strip().split() for email in emails if (
                email['email_type'] == 'int')]
        elif email_type == 'external':
            return [email['hb'].replace('\n', ' ').replace("SENT_END", "").strip().split() for email in emails if (
                email['email_type'] == 'ext' or email['email_type'] == 'mixed')]
        # agnostic to email type
        else:
            return [email['hb'].replace('\n', ' ').replace("SENT_END", "").strip().split() for email in emails]
    else:
        if email_type == 'internal':
            return [sent.strip().split() for email in emails for line in email['hb'].split('\n') for sent in line.split('SENT_END') if (
                len(sent) > 0 and email_type == 'int')]
        elif email_type == 'external':
            return [sent.strip().split() for email in emails for line in email['hb'].split('\n') for sent in line.split('SENT_END') if (
                len(sent) > 0 and (email_type == 'int' or email['email_type'] == 'mixed'))]
        else:
            return [sent.strip().split() for email in emails for line in email['hb'].split('\n') for sent in line.split('SENT_END') if len(sent) > 0]

def output_embeddings(mittens_df, filename, compress=False):
    if compress:
        mittens_df.to_csv(filename + '.gz', quoting=csv.QUOTE_NONE, header=False, sep=" ", compression='gzip')
    else:
        mittens_df.to_csv(filename, quoting=csv.QUOTE_NONE, header=False, sep=" ")
    return

#########################################################################
############# Helper Functions for Working with JSON Emails #############
#########################################################################
# Slightly different from get_recipients for spacespace emails
def get_recipients(msg):
    """
    Return a set of recipients of the current message.
    self is removed from list of recipients if in recipients
    All fields contain email addresses, not user IDs. From fields are visually just strings
    but checking just in case
    """
    sender = msg['From'][0] if type(msg['From']) == list else msg['From']
    return set(msg.get('To', []) + msg.get('Cc', []) + msg.get('Bcc', [])) - set([sender])

def slice_user_corpus(emails, train_mode):
    """
    Parameters
    ----------
    emails : list of dict
        A list of emails converted from JSON formats to dictionaries 
    train_mode : str
        One of 'annual', 'quarterly', 'all'
        Indicates how to chunk up emails - into quarters, years, or both
    Returns
    -------
    timekey2emails : dict
        Matches quarters or years to respective emails
    """
    timekey2emails = defaultdict(list)
    for email in emails:
        if train_mode == 'annual':
            timekey2emails[to_year(email['ActivityCreatedAt'], format='str')].append(email)
        elif train_mode == 'quarterly':
            timekey2emails[to_quarter(email['ActivityCreatedAt'], format='str')].append(email)
        elif train_mode == 'all':
            timekey2emails[to_year(email['ActivityCreatedAt'], format='str')].append(email)
            timekey2emails[to_quarter(email['ActivityCreatedAt'], format='str')].append(email)
    return timekey2emails

#########################################################################
############# Helper Functions for Working with Date Objects ############
#########################################################################

def to_quarter(date, format):
    """
    Return quarter of date in string
    """
    year, month = 0, 0
    if format == 'str':
        year = date[0:4]
        month = date[5:7]    
    elif format == 'datetime':
        year = date.year
        month = date.month
    quarter = ((int(month)-1) // 3) + 1
    timekey = str(year) + 'Q' + str(quarter)
    return timekey

def to_year(date, format):
    """
    Return year of date in string
    """
    if format == 'str':
        return date[0:4]
    elif format == 'datetime':
        return str(date.year)

def datetime_to_timekey(date, time_key):
    if time_key == 'year':
        return to_year(date, format='datetime')
    elif time_key == 'quarter':
        return to_quarter(date, format='datetime')

def is_month_before_equal(datetime1, datetime2):
    if datetime1.year < datetime2.year:
        return 1
    elif (datetime1.year == datetime2.year) and (datetime1.month <= datetime2.month):
        return 1
    else:
        return 0

def num_months_between_dates(datetime1, datetime2):
    return abs((datetime1.year - datetime2.year) * 12 + datetime1.month - datetime2.month)

def num_quarters_between_dates(datetime1, datetime2):
    return abs((datetime1.year - datetime2.year) * 12 + datetime1.month - datetime2.month) // 3

def num_years_between_dates(datetime1, datetime2):
    return abs(datetime1.year - datetime2.year)

def time_between_dates(datetime1, datetime2, time_key):
    if time_key == 'monthly':
        return num_months_between_dates(datetime1, datetime2)
    elif time_key == 'quarterly':
        return num_quarters_between_dates(datetime1, datetime2)
    elif time_key == 'annual':
        return num_years_between_dates(datetime1, datetime2)

#########################################################################
############## Helper Functions for Working with Dataframes #############
#########################################################################
def dict_to_df(index2rows, cols, index_name):
    """
    Parameters
    ----------
    index2rows : dict
        Dictionary mapping index to rows to be coverted
    cols : list
        List of column names of type str
    index : list
        List of index names
    Returns
    -------
    df : pd.DataFrame
        Constructed dataframe
    """
    if index2rows is None or len(index2rows) == 0:
        return None
    if len(index_name) == 1:
        df = pd.DataFrame.from_dict(index2rows, orient='index', columns=cols)
        df.index.name = index_name[0]
        df.sort_index(axis=0, inplace=True)
        return df
    else:
        df = pd.DataFrame.from_dict(index2rows, orient='index', columns=cols)
        df = pd.DataFrame(df, pd.MultiIndex.from_tuples(df.index, names=index_name))
        df.sort_index(axis=0, inplace=True)
        return df

#########################################################################
########### Helper Functions for Working with Embedding Output ##########
#########################################################################

def remove_empty_embeddings(embeddings_dir):
    """
    Removes all empty files in embeddings_dir that were produced when vocab size was 0.
    Parameters
    ----------
    embeddings_dir : str
        Full path to directory where embedding files are located
    """
    for file in os.listdir(embeddings_dir):
        mittens_file = os.path.join(embeddings_dir, file)
        if os.path.getsize(mittens_file) == 0:
            os.remove(mittens_file)
    return

def extract_company_embedding(company_embeddings_filename, tmp_dir, words):
    """
    Parameters
    ----------
    company_embeddings_filename : str
        File path of the company embeddings
    tmp_dir : str
        Path to the directory for gensim to output its tmp files in order to load embeddings into word2vec format
    words : list
        A list of strings for which to retrieve vectors
    Returns
    -------
    vecs : list
        A list of vectors of type numpy.ndarray that correspond to the list of words given as parameters
    """ 
    tmp_mittens = os.path.join(tmp_dir, "mittens_embeddings_all_word2vec.txt")
    word2vec_mittens_file = get_tmpfile(tmp_mittens)
    glove2word2vec(company_embeddings_filename, word2vec_mittens_file)
    model = KeyedVectors.load_word2vec_format(word2vec_mittens_file)
    vecs = []
    for w in words:
        if w in model.vocab:
            vecs.append(model.wv[w])
        else:
            vecs.append(np.nan)
            print('%s not in company embeddings' % w)
    return vecs

def word_similarity(model, w1, w2):
    """
    This is an auxilary function that allows for comparing one word to another word or multiple words
    If w1 and w2 are both single words, n_similarity returns their cosine similarity which is the same as 
    simply calling similarity(w1, w2)
    If w1 or w2 is a set of words, n_similarity essentially takes the mean of the set of words and then computes
    the cosine similarity between that vector mean and the other vector. This functionality is both reflected
    in its source code and has been verified manually.
    Parameters
    ----------
    model : KeyedVectors
        The model that contains all the words and vectors
    w1 : str or list
        The first word or word list to be compared
    w2 : str or list
        The second word or word list to be compared
    Returns
    -------
    float
        Cosine similarity between w1 and w2
    """
    if not isinstance(w1, list):
        w1 = [w1]
    if not isinstance(w2, list):
        w2 = [w2]
    w1 = [w for w in w1 if w in model.vocab]
    w2 = [w for w in w2 if w in model.vocab]
    if len(w1) == 0 or len(w2) == 0:
        return None
    return model.n_similarity(w1, w2)

def cossim_with_none(vec1, vec2, vec_format='sparse'):
    """
    Auxiliary function that calls cossim function to test if vectors are None to prevent erroring out.
    Parameters
    ----------
    vec1 : list of (int, float), gensim sparse vector format
    vec2 : list of (int, float), gensim sparse vector format
    format : str, optional
        Either sparse or dense. If sparse, vec1 and vec2 are in gensim sparse vector format; use cossim function from gensim.
        Otherwise, vec1 and vec2 are numpy arrays and cosine similarity is hand calculated
    Returns
    -------
    float
        Cosine similarity between vec1 and vec2
    """
    if not (isnull_wrapper(vec1) or isnull_wrapper(vec2)):
        if vec_format == 'sparse':
            return cossim(vec1, vec2)
        elif vec_format == 'dense':
            if len(vec1) == 0 or len(vec2) == 0:
                return None
            return np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))
        else:
            raise ValueError()
    return None

def calculate_pairwise_cossim(col1, col2=None, reference=False, reference_group=None, anon_ids=None, vec_format='sparse'):
    """
    Calculates averaged cosine similarity of every row vector in col1 with every other row vector in col2.
    If no col2 is provided, cosine similarity of every row vector with every other row vector in col1 is calculated.
    The two columns should have equal length.
    Parameters
    ----------
    col1 : pd.Series
        A column where each row is a sparse word vector (BoW format that gensim code is written for).
    col2 : pd.Series, optional
        A column where each row is a sparse word vector (BoW format that gensim code is written for).
    reference : bool, optional
        Indicator variable for whether filtering for reference groups is needed
    reference_group : pd.Series, optional
        If filtering for reference groups, a list containing reference group members for every employee in col1
    anon_ids : pd.Series, optional
        If filtering for reference groups, a list containing anon_ids for every employee in col1
    Returns
    -------
    results : list
        A list where the ith element is the averaged cosine similarity between the ith vector in col1 and every vector
        in col2 for which i != j. If no col2 is provided, a list where the ith element is the averaged cosine similarity
        between the ith vector in col1 and every other vector in col1 is returned.
    """
    vectors1 = col1.tolist()
    vectors2 = col2.tolist() if col2 is not None else col1.tolist()
    reference_group = reference_group.tolist() if reference else None
    anon_ids = anon_ids.tolist() if anon_ids is not None else None
    results = list()
    for i in range(len(vectors1)):
        total_sim = []
        if not isnull_wrapper(vectors1[i]):
            for j in range(len(vectors2)):
                if i != j and not isnull_wrapper(vectors2[j]):
                    # filter out any np.nans as our reference group
                    if not reference or (type(reference_group[i]) == set and anon_ids[j] in reference_group[i]):
                        if vec_format == 'sparse':
                            total_sim.append(cossim(vectors1[i], vectors2[j]))
                        elif vec_format == 'dense':
                            total_sim.append(np.dot(vectors1[i], vectors2[j])/(np.linalg.norm(vectors1[i]) * np.linalg.norm(vectors2[j])))
                        else:
                            raise ValueError()
            results.append(mean(total_sim) if len(total_sim) > 0 else None)
        else:
            results.append(None)
    return results

def isnull_wrapper(x):
    r = pd.isnull(x)
    if type(r) == bool:
        return r
    return r.any()

def vector_mean(col):
    """
    Calculate vector means of row vectors
    Parameters
    ----------
    col : pd.Series
        The column to be averaged
    Returns
    -------
    np.array
        A vector that is the numerical average of all vectors in col
    """
    return np.array(col[col.notna()].tolist()).mean(axis=0)

def extract_hr_survey_df(survey_hr_file, user_qualtrics_file, users_file, perf_likert_file, perf_percentage_file):
    """
    Loads various datasets to prepare survey and hr data for merging with embeddings df.
    Returns
    -------
    survey_hr_df : pd.DataFrame
        Dataframe indexed by user_id
    """
    perf_likert_df = pd.read_csv(perf_likert_file)
    perf_percentage_df = pd.read_csv(perf_percentage_file)
    perf_likert_df = perf_likert_df[['UID', '2019 Perf_Type(Rating)', '2020 Perf_Type(Rating)']]
    perf_percentage_df = perf_percentage_df[['UID', '2019_Perf_Type(Percentage)', '2020_Perf_Type(Percentage)']]
    perf_likert_df.columns = ["UID", "perf_rating_2019", "perf_rating_2020"]
    perf_percentage_df.columns = ["UID", "perf_percentage_2019", "perf_percentage_2020"]
    
    user_qualtrics_df = pd.read_csv(user_qualtrics_file)
    user_qualtrics_df = user_qualtrics_df.merge(perf_likert_df, on='UID', how='left').merge(perf_percentage_df, on="UID", how='left')

    survey_hr_df = pd.read_csv(survey_hr_file)
    survey_hr_df = survey_hr_df.merge(user_qualtrics_df, left_on='uid', right_on='UID', how='left')
    # we lose two employees whose emails are not included in the crawled email data
    email2uid = {}
    with open(users_file, encoding='utf-8') as f:
        for line in f:
            user = json.loads(line)
            for e in user['Emails']:
                email2uid[e] = user['UserId']

    survey_hr_df = survey_hr_df[survey_hr_df['Email'].isin(email2uid.keys())]
    survey_hr_df['user_id'] = survey_hr_df['Email'].apply(lambda e : email2uid[e])
    survey_hr_df.set_index('user_id', inplace=True)
    return survey_hr_df

def extract_variables_from_file(file):
    """ 
    Extract relevant information from name of embedding file, with format: {}(_{})?_(internal|external).txt
    """
    file_chunks = file[0:-4].split('_')
    usr = file_chunks[0]
    time_key = file_chunks[1] if len(file_chunks) == 3 else None
    return (usr, time_key)

def project(word, dimension):
    """
    Returns the scalar projection of word on dimension. Word and dimension are both assumed to be vectors
    """
    return np.dot(word, dimension)/ np.linalg.norm(dimension)

def drop(u, v):
    """
    Returns the component in u that is orthogonal to v, also known as the vector rejection of u from v
    """
    return u - v * u.dot(v) / v.dot(v)

def remove_frequency(company_embeddings, embedding_dim):
    """
    Remove frequency dimension from company embeddings (assumed to be the top dimension)
    Parameters
    ----------
    company_embeddings : dict
        A dictionary mapping words to their embeddings
    embedding_dim : int
        The dimension of word vectors. Used to determine the number of components that go into PCA.
    Returns
    -------
    new_company_embeddings : dict
        A dictionary mapping words to their embeddings, where embeddings are demeaned and first PCA
        dimension hypothesized to represent frequency dimension is removed
    """
    vectors = np.array([v for k, v in company_embeddings.items()])
    miu = np.mean(vectors, axis=0)
    demeaned_vectors = vectors - miu
    pca = PCA(n_components = embedding_dim)
    pca.fit(demeaned_vectors)
    frequency_dim = pca.components_[0]
    new_company_embeddings = {k : drop(v-miu, frequency_dim) for k, v in company_embeddings.items()}
    return new_company_embeddings

def doPCA(words_start, words_end):
    """
    Performs PCA on differences between pairs of words and returns the first component
    Based on function doPCA in Bolukbasi et al. (2016) source code at https://github.com/tolga-b/debiaswe/blob/master/debiaswe/we.py
    Parameter
    ---------
    words_start : list
        List of hashed words at one end of interested dimension
    words_end: list
        List of hashed words at the other end of dimension
    Returns
    -------
    ndarray
        First component of PCA of differences between pairs of words
    """
    matrix = []
    for i in range(len(words_start)):
        center = (words_start[i] + words_end[i])/2
        matrix.append(words_end[i] - center)
        matrix.append(words_start[i] - center)
    matrix = np.array(matrix)
    # cannot have more components than the number of samples
    num_components = len(words_start)*2
    pca = PCA(n_components = num_components)
    pca.fit(matrix)
    return pca.components_[0]

def build_dimension(words_start, words_end):
    """
    This method builds a dimension defined by words at separate end of a dimension.
    Multiple methods exist in previous literature when building such a dimension.
    1) Kozlowski et al. (2019) averages across differences between different word pairs, noted to be interchangeable with averaging words on each side of the dimension and
    then taking the difference between averages. They are empirically verified to be identical.
    2) Bolukbasi et al. (2016) defines gender direction using a simple difference between man and woman in the corresponding tutorial. In the same tutorial, 
    racial direction is defined as difference between two clusters of words that are each sum of the embeddings of its corresponding dimensions
    normalized by the L2 norm. Wang et al. (2020) note that normalization is unnecessary. If unnormalized, this method should be equivalent to #3.
    3) Bolukbasi et al. (2016) defines gender direction also by taking the differences across multiple pairs, doing PCA on these differences, and 
    taking the first component as the gender direction.
    Parameter
    ---------
    words_start : list
        List of hashed words at the positive end of the dimension, where positive implies more likely to affect identification positively
    words_end: list
        List of hashed words at the other end of dimension
    Returns
    -------
    (mean_dim, pca_dimension) : 2-tuple of numpy vector
        Two vector that represents the dimension of interest calculated using method #1 and #3.
    """
    assert len(words_start) == len(words_end)
    differences = [(np.array(words_start[i]) - np.array(words_end[i])) for i in range(len(words_start)) if not np.isnan(words_start[i]).any() and not np.isnan(words_end[i]).any()]
    mean_dim = np.array(differences).mean(axis=0)
    pca_dim = doPCA(words_start, words_end)
    if project(words_start[0], pca_dim) < 0:
        # convention used in the current script is that words_start should represent the positive dimension
        pca_dim = pca_dim * -1
    return (mean_dim, pca_dim)

