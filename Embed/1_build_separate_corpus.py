#!/usr/bin/python3
"""
Builds corpus using hashed emails in format expected by GloVe.
Documents and documents only are separated by newline characters. All tokens are separated by spaces.
"""
import os
import ujson as json
import pandas as pd
from datetime import datetime

home_dir = "/ifs/projects/amirgo-identification/"
email_dir = os.path.join(home_dir, "email_data")
email_file = os.path.join(email_dir, 'MessagesHashed.jsonl')
activity_file = os.path.join(email_dir, 'Activities.json')
corpus_file_american = 'GloVe-master/corpus_american.txt'
corpus_file_indian = 'GloVe-master/corpus_indian.txt'
users_file = os.path.join(email_dir, 'Users.json')
activity_file = os.path.join(email_dir, 'Activities.json')
survey_dir = os.path.join(home_dir, "survey_hr_data")
user_qualtrics_file = os.path.join(survey_dir, "UsersQualtrics.csv")
analyses_data_dir = "/ifs/gsb/amirgo/spacespace/spacespace/Coco/analyses_data/"
survey_filename = os.path.join(analyses_data_dir, "preprocessed_survey_hr.csv")

total_emails, indian_emails, american_emails = 0, 0, 0

def read_emails():
    sid2user_id, email2uid = dict(), dict()
    with open(activity_file, encoding='utf-8') as f:
        for line in f:
            activity = json.loads(line)
            sid2user_id[activity['MailSummarySid']] = activity['UserId']

    with open(users_file, encoding='utf-8') as f:
        for line in f:
            user = json.loads(line)
            for e in user['Emails']:
                email2uid[e] = user['UserId']

    user_qualtrics_df = pd.read_csv(user_qualtrics_file)
    survey_hr_df = pd.read_csv(survey_filename).merge(user_qualtrics_df, left_on='uid', right_on='UID', how='left')
    # we lose two employees whose emails are not included in the crawled email data    
    survey_hr_df = survey_hr_df[survey_hr_df['Email'].isin(email2uid.keys())]
    survey_hr_df['user_id'] = survey_hr_df['Email'].apply(lambda e : email2uid[e])
    survey_hr_df.set_index('user_id', inplace=True)
    surveyed_users = survey_hr_df.index.values
    
    output_file_indian = open(corpus_file_indian, 'w')
    output_file_american = open(corpus_file_american, 'w')

    with open(email_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print("Processed {} emails".format(i))
            global total_emails, indian_emails, american_emails
            total_emails += 1
            email = json.loads(line)
            lang = email['l']
            body = email['hb'].strip()
            if len(body) > 0:
                if lang[0] == "__label__en" and (lang[1] > 0.5 or len(email['liwc']) > 0):
                    body = body.replace("\n", " ").replace("SENT_END", "")
                    # Code in GloVe-master (get_word function in common.c) seems to imply that newline characters are expected to be separate from text
                    user_id = sid2user_id[email['sid']]
                    if user_id in surveyed_users:
                        location = survey_hr_df.loc[user_id, 'Work Country'] 
                        if location == 'India':
                            output_file_indian.write(body + ' \n ')
                            indian_emails += 1
                        else:
                            output_file_american.write(body + ' \n ')
                            american_emails += 1
    output_file_indian.close()
    output_file_american.close()
    return

if __name__ == '__main__':
    starttime = datetime.now()
    read_emails()
    print("""Out of {} emails processed, {} emails were written.\n Americans: {} emails. Indians: {} emails.\n Finished at {}, with a duration of {}.\n""".format(
        total_emails, indian_emails+american_emails, american_emails, indian_emails, str(datetime.now()), str(datetime.now() - starttime)))
