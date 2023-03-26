from hash import *
from test_hash_data import *

def test_clean():
    """
    Unit test of the body_to_hash function
    """
    for i in range(0,len(test_emails)):
        body = test_emails[i]
        hashed_body, clean_body, _ = body_to_hash(body)

        assert clean_body == correct_clean_body[i], \
            "Email #%d cleaning error.\nOriginal Email:\n%s\nCleaned Email:\n%s\nExpected Email:\n%s\n" % (i, body, clean_body, correct_clean_body[i])
    print('Cleaned emails match expected emails')
    return

def test_hash():
    """
    Unit test of the body_to_hash function
    """
    for i in range(0,len(test_emails)):
        body = test_emails[i]
        hashed_body, _, lang = body_to_hash(body, return_liwc=False)

        assert hashed_body == correct_hashed_body[i], \
            "Email #%d hashing error.\nOriginal Email:\n%s\nHashed Email:\n%s\nExpected Email:\n%s\n" % (i, body, hashed_body, correct_hashed_body[i])
    print('Hashed emails match expected emails')
    return

def test_unhash():
    """
    Unit test of the body_to_hash function
    """
    hash2word = {}
    hash2word['SENT_END'] = 'SENT_END'
    hashed_bodies = []
    for i in range(0, len(test_emails)):
        body = test_emails[i]
        hashed_body, word2hash, lang = body_to_hash(body)
        hash2word.update({v : k for k, v in word2hash.items()})
        hashed_bodies.append(hashed_body)

    for i in range(0, len(hashed_bodies)):
        hash_body = hashed_bodies[i]
        clean_body = correct_clean_body[i]
        unhashed_lines = []
        for line in hash_body.split('\n'):
            if len(line.strip()) > 0:
                unhashed_lines.append(' '.join([hash2word[t] for t in line.split(' ')]))
        unhashed_body = '\n'.join(unhashed_lines)
        if len(unhashed_body) > 0:
        	unhashed_body += '\n'
        unhashed_body = re.sub(r'\d+', '', unhashed_body)
        clean_body = re.sub(r'\d+', '', clean_body)
        assert unhashed_body == clean_body, \
            "Email #%d unhashing error.\nUnhashed Email:\n%s\nClean Email:\n%s\n" % (i, unhashed_body, clean_body)
    print('Unhashing emails match expected emails')
    return
    
def test_lang():
    for i in range(0,len(multi_lang_emails)):
        body = multi_lang_emails[i]
        hashed_body, _, lang = body_to_hash(body, return_liwc=False)
        is_eng = lang[0] == '__label__en' and lang[1] > 0.5
        assert is_eng == correct_lang[i], "Email #%d language detection error.\nDetected: %s\nExpected: %s\n" % (i, is_eng, correct_lang[i])
    print('Language detection works as expected')
    return

def test_liwc():
    read_liwc_dictionary('LIWC2007dictionary poster.csv')
    for i in range(0, len(test_emails)):
        body = test_emails[i]
        hashed_body, word2hash, lang, liwc = body_to_hash(body, return_liwc=True)
        assert liwc == correct_liwc[i], "Email #%d liwc error.\nGot: %s\nExpected: %s\n" % (i, liwc, correct_liwc[i])
    print('LIWC works as expected')


if __name__ == '__main__':
    test_hash()
    test_unhash()
    test_lang()
    test_liwc()