#!/usr/bin/python3
"""
Builds corpus using hashed emails in format expected by GloVe.
Documents and documents only are separated by newline characters. All tokens are separated by spaces.
"""
import os
import ujson as json

total_emails = 0
non_english = 0
english = 0
ling_thres = 0.95

email_dir = "/ifs/projects/amirgo-identification/email_data/"
out_dir = "/ifs/gsb/amirgo/spacespace/spacespace/Coco/Embed/GloVe-master/"
email_file = os.path.join(email_dir, 'MessagesHashed.jsonl')
corpus_file = os.path.join(out_dir, 'corpus_high_prob_eng_{}.txt'.format(str(ling_thres).replace(".", "")))

def read_emails(in_file, out_file):
    output_file = open(out_file, 'w')
    with open(in_file, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0:
                print("Processed {} emails".format(i))
            global total_emails, english, non_english
            total_emails += 1
            email = json.loads(line)
            lang = email['l']
            body = email['hb'].strip()
            if len(body) > 0:
                # original - if lang[0] == "__label__en" and (lang[1] > 0.5 or len(email['liwc']) > 0):
                if lang[0] == "__label__en" and (lang[1] > ling_thres or len(email['liwc']) > 5):
                    body = body.replace("\n", " ").replace("SENT_END", "")
                    # Code in GloVe-master (get_word function in common.c) seems to imply that newline characters are expected to be separate from text
                    output_file.write(body + ' \n ')
                    english += 1
                elif len(lang[0]) > 0:
                    non_english += 1
    output_file.close()
    return

if __name__ == '__main__':
    read_emails(email_file, corpus_file)
    print("""Out of {} emails processed, {} emails were non_empty.\n
        {} English emails were written to corpus.txt. {} emails non-English emails are discarded.""".format(total_emails, english+non_english, english, non_english))
