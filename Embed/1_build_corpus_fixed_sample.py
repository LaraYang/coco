#!/usr/bin/python3
"""
Builds corpus using hashed emails in format expected by GloVe.
Documents and documents only are separated by newline characters. All tokens are separated by spaces.
"""
import os
import ujson as json
from numpy import random
from collections import defaultdict
from datetime import datetime
email_dir = "/ifs/projects/amirgo-identification/email_data/"
email_file = os.path.join(email_dir, 'MessagesHashed.jsonl')
activity_file = os.path.join(email_dir, 'Activities.json')
corpus_file = os.path.join('/ifs/gsb/amirgo/spacespace/spacespace/Coco/Embed/GloVe-master/corpus_fixed_sample.txt')
total_emails = 0
non_english = 0
english = 0

def read_emails():
    output_file = open(corpus_file, 'w')
    sid2user_id = dict()
    with open(activity_file, encoding='utf-8') as f:
        for line in f:
            activity = json.loads(line)
            sid2user_id[activity['MailSummarySid']] = activity['UserId']

    user_id2emails = defaultdict(list)
    with open(email_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print("Processed {} million emails".format(i/1000000))
            global total_emails, english, non_english
            total_emails += 1
            email = json.loads(line)
            lang = email['l']
            body = email['hb'].strip()
            if len(body) > 0:
                if lang[0] == "__label__en" and (lang[1] > 0.5 or len(email['liwc']) > 0):
                    body = body.replace("\n", " ").replace("SENT_END", "")
                    # Code in GloVe-master (get_word function in common.c) seems to imply that newline characters are expected to be separate from text
                    user_id2emails[sid2user_id[email['sid']]].append(body+' \n ')
                    english += 1
                elif len(lang[0]) > 0:
                    non_english += 1
    # approximate average number of emails per user: 22522067 million non-empty, English emails / 1700, 13248
    sample = 13248

    num_files_written = 0
    for user, emails in user_id2emails.items():
        # downsampling + upsampling
        chosen = random.choice(emails, sample)
        for c in chosen:
            output_file.write(c)
            num_files_written += 1
    print("Sampled {} emails from all users. Wrote {} emails in total".format(sample, num_files_written))
    output_file.close()
    return

if __name__ == '__main__':
    starttime = datetime.now()
    read_emails()
    print("""Out of {} emails processed, {} emails were non_empty.\n {} English emails were written. {} emails non-English emails are discarded.\n Finished at {}, with a duration of {}.\n""".format(
        total_emails, english+non_english, english, non_english, str(datetime.now()), str(datetime.now() - starttime)))
