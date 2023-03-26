#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    3e_output_effort.py test|actual
Producing files that measures effort variables
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
from sklearn import preprocessing
from datetime import datetime, timezone
import us
import pytz

exclude_external = False
home_dir = "/ifs/projects/amirgo-identification/"
email_dir = os.path.join(home_dir, "email_data/")
email_file = os.path.join(email_dir, 'MessagesHashed.jsonl')
users_file = os.path.join(email_dir, 'Users.json')
activity_file = os.path.join(email_dir, 'Activities.json')

survey_dir = os.path.join(home_dir, "survey_hr_data")
user_qualtrics_file = os.path.join(survey_dir, "UsersQualtrics.csv")
perf_percentage = os.path.join(survey_dir, "perf_rating_percentages.csv")
perf_likert = os.path.join(survey_dir, "perf_rating_likert.csv")

analyses_data_dir = "/ifs/gsb/amirgo/spacespace/spacespace/Coco/analyses_data/"
survey_filename = os.path.join(analyses_data_dir, "preprocessed_survey_hr.csv")

output_dir = os.path.join(home_dir, "email_idtf_data")
quarterly_output_filename = os.path.join(output_dir, "effort_quarterly.csv") if exclude_external else os.path.join(output_dir, "effort_quarterly_all.csv")

year_colname, quarter_colname = 'year', 'quarter'

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
def scale(x):
    """
    This scale function provides the same result as R's scale. sklearn's preprocessing.scale is slightly different as it uses N instead of N-1
    in calculating standard deviation 
    """
    return (x-np.mean(x))/np.std(x, ddof=1)

def read_raw_counts(activity_file, email_file, hr_df, test_mode=False):
    """
    The main workhorse function for obtaining number of messages sent post work as a measure of effort
    Parameters
    ----------
    activity_file : str
        The full filepath that contains all email metadata, where each line is a JSON object that represents one email
    email_file : str
        The full filepath that contains all email content, where each line is a JSON object that represents one email
    hr_df : pd.DataFrame
        Used for departmental standardizing of effort variables
    test_mode : optional, bool
        Used for quick processing when processing
    Returns
    -------
    tuple
        A tuple of user-level, annual, and quarterly dataframes
    """
    usr2counts, usr_year2counts, usr_quarter2counts = defaultdict(lambda : [0, 0, 0, 0, set(), set(), []]), defaultdict(lambda : [0, 0, 0, 0, set(), set(), []]), defaultdict(lambda :[0, 0, 0, 0, set(), set(), []])
    sid2activity = {}
    thread2times = defaultdict(list)
    email2user_id = {}
    user_id2state = hr_df['Work State'].to_dict()
    states = hr_df['Work State'].to_list()
    state2timezone = {}
    for state in set(states):
        us_state = us.states.lookup(state)
        if us_state:
            # some states span more than one time zone; if so, choose the first one
            state2timezone[state] = us_state.time_zones[0]
    state2timezone['British Columbia'] = 'America/Vancouver'
    state2timezone['Ontario'] = 'America/Toronto'
    state2timezone['Karnataka'] = 'Asia/Kolkata'
    state2timezone['Gujarat'] = 'Asia/Kolkata'
    with open(activity_file, encoding='utf-8') as f:
        for line in f:
            activity = json.loads(line)
            sid2activity[activity['MailSummarySid']] = activity
            thread2times[activity['ThreadId'][-26:]].append(activity['ActivityCreatedAt'])
    for thread, times in thread2times.items():
        thread2times[thread] = sorted(times)

    with open(users_file, encoding='utf-8') as f:
        for line in f:
            user = json.loads(line)
            for e in user['Emails']:
                email2user_id[e] = user['UserId']

    with open(email_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if test_mode and i > 10000:
                break
            email = json.loads(line)
            lang = email['l']
        
            activity = sid2activity[email['sid']]
            user = activity['UserId']
            quarter = to_quarter(activity['ActivityCreatedAt'], format='str')
            num_toks = len(email['hb'].replace('\n', ' ').replace("SENT_END", "").strip().split())
            activity = sid2activity[email['sid']]
            recipients = get_recipients(activity)

            if exclude_external:
                # excluding external emails
                pure_internal = True
                for r in recipients:
                    domain = r.split('@')[1]
                    if domain not in collabera_hashes:
                        pure_internal = False
                        break
                if not pure_internal:
                    continue
            # convert global UTC timezone to local timezone
            date = datetime.strptime(activity['ActivityCreatedAt'], '%Y-%m-%dT%H:%M:%S.000Z')
            timezoned_date = date.replace(tzinfo=timezone.utc).astimezone(pytz.timezone(state2timezone[user_id2state[user]])) if user in user_id2state else date
            weekday = timezoned_date.weekday()
            hour = timezoned_date.hour

            usr_quarter2counts[(user, quarter)][0] += num_toks
            usr_quarter2counts[(user, quarter)][1] += 1
            if (weekday == 5) or (weekday == 6):
                usr_quarter2counts[(user, quarter)][2] += 1
                usr_quarter2counts[(user, quarter)][4].add(timezoned_date.strftime('%Y-%m-%d'))
            elif (hour < 8) or (hour > 17):
                usr_quarter2counts[(user, quarter)][3] += 1
            for r in recipients:
                if r in email2user_id:
                    peer = email2user_id[r]
                    if user != peer:
                        usr_quarter2counts[(user, quarter)][5].add(peer)
            curr_time_index = thread2times[activity['ThreadId'][-26:]].index(activity['ActivityCreatedAt'])
            if curr_time_index > 0:
                response_time = date - datetime.strptime(thread2times[activity['ThreadId'][-26:]][curr_time_index-1], '%Y-%m-%dT%H:%M:%S.000Z')
                usr_quarter2counts[(user, quarter)][6].append(response_time.total_seconds())

    for key in list(usr_quarter2counts):
        usr_quarter2counts[key][4] = len(usr_quarter2counts[key][4])
        # this ensures peers are both incoming and outgoing ties
        # peers originally only contain outgoing ties; we go through the list once to make sure incoming ties are also accounted for
        for peer in usr_quarter2counts[key][5]:
            usr_quarter2counts[(peer, key[1])][5].add(key[0])
        if len(usr_quarter2counts[key][6]) > 0:
            usr_quarter2counts[key][6] = sum(usr_quarter2counts[key][6])/len(usr_quarter2counts[key][6])
        else:
            usr_quarter2counts[key][6] = None
    
    for key in list(usr_quarter2counts):
        peers = usr_quarter2counts[key][5]
        curr_working_weekends = usr_quarter2counts[key][4] if type(usr_quarter2counts[key][4]) == int else 0
        peer_num_messages_weekend = [usr_quarter2counts[(p, key[1])][2] for p in peers] + [usr_quarter2counts[key][2]]
        peer_num_messages_post_work = [usr_quarter2counts[(p, key[1])][3] for p in peers] + [usr_quarter2counts[key][3]]
        peer_num_working_weekends = [usr_quarter2counts[(p, key[1])][4] if type(usr_quarter2counts[(p, key[1])][4]) == int else 0 for p in peers] + [curr_working_weekends]
        
        usr_quarter2counts[key].append(scale(peer_num_messages_weekend)[-1])
        usr_quarter2counts[key].append(scale(peer_num_messages_post_work)[-1])
        usr_quarter2counts[key].append(scale(peer_num_working_weekends)[-1])

    cols = (['num_tokens', 'num_messages', 'num_messages_weekend', 'num_messages_post_work', 'num_working_weekends', 'network_peers', 'avg_response_time',
    'peer_standardized_num_messages_weekend', 'peer_standardized_num_messages_post_work', 'peer_standardized_num_working_weekends'])
    usr_quarter2counts_df = hr_df.join(dict_to_df(usr_quarter2counts, cols, index_name=['user_id', quarter_colname]))
    
    usr_quarter2counts_df['num_working_weekends'] = usr_quarter2counts_df['num_working_weekends'].apply(lambda x : x if type(x) == int else 0)
    for var in ['num_messages_weekend', 'num_messages_post_work', 'num_working_weekends']:
        usr_quarter2counts_df['department_standardized_'+var] = usr_quarter2counts_df.groupby(['quarter', 'Department'])[var].apply(scale)
    return usr_quarter2counts_df

if __name__ == '__main__':
    starttime = datetime.now()
    test = False
    try:
        test = sys.argv[1].lower() == 'test'
    except IndexError as error:
        pass
    if test:
        quarterly_output_filename = os.path.join(output_dir, "test_effort_quarterly.csv")

    sys.stderr.write('Reading HR and Survey data at %s.\n' % datetime.now())
    hr_df = extract_hr_survey_df(survey_filename, user_qualtrics_file, users_file, perf_likert, perf_percentage)

    sys.stderr.write('Loading corpus counts at %s.\n' % datetime.now())
    usr_quarter2counts_df = read_raw_counts(activity_file, email_file, hr_df, test)

    sys.stderr.write('Outputting dataframe at %s.\n' % datetime.now())
    if usr_quarter2counts_df is not None: usr_quarter2counts_df.to_csv(quarterly_output_filename)
    
    sys.stderr.write("Finished outputting measures at %s, with a duration of %s.\n"
        % (str(datetime.now()), str(datetime.now() - starttime)))
