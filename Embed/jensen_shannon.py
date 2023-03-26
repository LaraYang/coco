from datetime import datetime
import os
import re
import glob
import sys
import random
import logging
import pickle
from operator import itemgetter
from collections import defaultdict, Counter
import numpy as np
import csv

def node_term_count_distribution(nodes, usr_quarter2liwc, quarter):
    """
    Sum liwc counts across all nodes and convert the overall dictionary into 
    a distribution.
    nodes : list
        List of node names
    usr_quarter2liwcs: dict
        Maps (user_id, quarter) tuples to a Counter of all LIWC counts of said user
    quarter : str
        A string that represents the focal quarter
    """
    worddict = defaultdict(int)
    for n in nodes:
        liwc = usr_quarter2liwc[n, quarter]
        for cat, count in liwc.items():
            worddict[cat] += count
    dist = counts2dist(worddict)
    return dist

def get_term_count_distribution(all_liwc):
    """
    Sum liwc counts across all dictionaries in all_liwc and convert the overall dictionary into a probability distribution.
    Parameter
    ---------
    all_liwc : list of dict of {str : int}
        A list of LIWC categories counts
    Returns
    -------
    dist : dict of {str : float}
        A probability distribution of LIWC categories
    """
    worddict = defaultdict(int)
    for liwc in all_liwc:
        for cat, count in liwc.items():
            worddict[cat] += count
    dist = counts2dist(worddict)
    return dist

def counts2dist(countdict):
    """
    Turns a dictionary of counts to a dictionary of probabilities.
    Parameter
    ---------
    countdict : dict of {str : int}
        A dictionary of counts
    Returns
    -------
    dict of {str : float}

    """
    total = float(sum(countdict.values()))
    return {key:val/total for key, val in countdict.items()}

def jensen_shannon(f, g):
    """
    Provides the jensen_shannon distance between two probability distributions f and g. See https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence.
    Parameters
    ----------
    f : dict of {str : int}
        A liwc distribution mapping keys to their probabilities
    g : dict of {str : int}
        A liwc distribution mapping keys to their probabilities
    Returns
    -------
    float
        JS distance
    """
    if f is None or g is None or len(f) == 0 or len(g) == 0:
        return None
    vocab = sorted(set(f.keys()) | set(g.keys()))
    p = np.zeros(len(vocab))
    q = np.zeros(len(vocab))
    for i, w in enumerate(vocab):
        p[i] = f.get(w, 0.0)
        q[i] = g.get(w, 0.0)
    pq = (p + q) / 2.0                
    a = 0.5 * kl(p, pq)
    b = 0.5 * kl(q, pq)
    return np.sqrt(a + b)

def kl(p, q):
    """
    Provides the Kullback-Leibler divergence between two probability distributions p and q
    Parameters
    ----------
    p : np.array
        An array that represents a probability distribution
    q : np.array
        An array that represents a probability distribution
    Returns
    -------
    float
        KL divergence
    """
    return np.sum(p * safelog2(p/q))

def safelog2(x):
    """
    Provides base-2 logarithm of float x while handling errors.
    """
    with np.errstate(divide='ignore'):
        x = np.log2(x)
        x[np.isinf(x)] = 0.0
        return x

def js2cf(dist):
    """
    Converts JS distance to cultural fit measure.
    Parameters
    ----------
    dist : float
        JS distance
    Returns
    -------
    float 
        Cultural fit measure
    """
    if dist == 0:
            dist = 0.0000000001
    if dist:
        return -np.log(dist)
    return None
