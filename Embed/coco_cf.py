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
email2user_id = {}
quarter2usr2sent_liwcs, quarter2usr2received_liwcs = defaultdict(lambda : defaultdict(Counter)), defaultdict(lambda : defaultdict(Counter))

domain_hash = {
    'collabera.com':                     '509c8f6b1127bceefd418c023533d653', 
    'collaberainc.mail.onmicrosoft.com': 'ec5b67548b6ec06f1234d198efec741e', 
    'collaberainc.onmicrosoft.com':      '86160680578ee9258f097a67a5f25af9', 
    'collaberasp.com':                   '6bf3934d19f1acf5b9295b63e0e7f66e', 
    'g-c-i.com':                         '3444d1f7d5e46443080f2d069e41a10c'}
collabera_hashes = set([v for k, v in domain_hash.items()])

lines_to_test = 2000

def get_documents(test_mode):
    """
    Reads all LIWC dictionaries from emails to populate sent and received LIWC dictionaries
    Parameter
    ---------
    test_mode : bool
        If true, restrict nuber of emails read
    """
    global email2user_id, quarter2usr2received_liwcs, quarter2usr2sent_liwcs
    sid2liwc = defaultdict(dict)
    
    with open(users_file, encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            if test_mode and i > lines_to_test: break
            user = json.loads(line)
            for e in user['Emails']:
                email2user_id[e] = user['UserId']
    
    with open(email_file, encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            if test_mode and i > lines_to_test: break
            email = json.loads(line)
            sid2liwc[email['sid']] = email['liwc']

    edges = defaultdict(list)
    with open(activity_file, encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
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
                liwc = sid2liwc[activity["MailSummarySid"]]
                if liwc is not None:
                    for r in recipients_userids:
                            quarter2usr2received_liwcs[quarter][r].update(Counter(liwc))
                    quarter2usr2sent_liwcs[quarter][sender_userid].update(Counter(liwc))
                
    return

def get_cultural_fit(quarter, usr2sent_liwcs, usr2received_liwcs):
    """
    Calculate the JS distance between an individual's sent emails and received emails for a given quarter and
    converts JS into CF. This version of CF is the original CF measure used.
    Parameters
    ----------
    quarter : str
        The quarter for which cultural fit is being calculated. This is needed only to mark the returned dictionary
        with the correct quarter so that all the parallel results can be joined together effectively.
    usr2sent_liwcs : defaultdict of {str : Counter}
        A dictionary that maps users to a list of LIWC dictionaries in messages they sent. Note the key difference
        between the type of this argument and that in corpcorp_cf.py and spsp_cf.py. In the latter two datasets,
        this argument is of type {str : list of dict}. Given the number of emails in this dataset,
        we are itereratively updating the sent LIWCs to be a single dictionary of counts instead of a list of dictionaries to save RAM.
    usr2received_liwcs : defaultdict of {str : Counter}
        A dictionary that maps users to a list of LIWC dictionaries in messages they received. Same as above in 
        argument type.
    Returns
    -------
    usr_quarter2cf : defaultdict of {(str, str) : float}
    """
    
    usr_quarter2cf = defaultdict(float)
    for u, liwcs in usr2sent_liwcs.items():
        usr_quarter2cf[(u, quarter)] = js2cf(jensen_shannon(counts2dist(liwcs), counts2dist(usr2received_liwcs[u])))
    return usr_quarter2cf
    
def get_quarterly_cultural_fit(test_mode):
    """
    Computes cultural fit for all quarters by running computations for all quarters in parallel.
    Parameters
    ----------
    test_mode : bool, optional
        Whether we are testing the code or running for production
    """
    get_documents(test_mode)
    file_name = 'coco_cf'
    if test_mode:
        file_name += "_test"
        
    pool = multiprocessing.Pool(processes = num_cores)
    results = [pool.apply_async(get_cultural_fit, args=(timekey, usr2sent_liwcs, quarter2usr2received_liwcs[timekey])) for timekey, usr2sent_liwcs in quarter2usr2sent_liwcs.items()]
    pool.close()
    pool.join()
    usr_quarter2cf = defaultdict(float)
    for r in results:
        usr_quarter2cf.update(r.get())
    df = dict_to_df(usr_quarter2cf, ["vanilla_cf"], index_name=['user_id', 'quarter'])
    df.round(5).to_csv(os.path.join(output_dir, file_name + '.csv'), index=True)
    sys.stderr.write("Finished outputting cultural fit at %s.\n" % str(datetime.now()))
    return

if __name__ == '__main__':
    starttime = datetime.now()
    test_mode = False
    try:
        test_mode = sys.argv[1].lower() == 'test'
    except IndexError as error:
        pass

    sys.stderr.write('Started Processing at %s.\n' % datetime.now())
    get_quarterly_cultural_fit(test_mode)
    sys.stderr.write('Finished Processing at %s with a duration of %s.\n' % (datetime.now(), str(datetime.now()-starttime)))


