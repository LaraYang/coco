#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    python3 hash.py
"""
import os
import sys
from hashlib import md5
from collections import Counter
import random
import contractions
from operator import add, itemgetter
import re
from talon.signature.bruteforce import extract_signature
# for language detection; install via pip install fasttext and download model from https://fasttext.cc/docs/en/language-identification.html
import fasttext
model = fasttext.load_model('lid.176.ftz')


import nltk
from nltk.stem import *
from nltk import TweetTokenizer
import csv

stemmer = PorterStemmer()
tokenizer = TweetTokenizer()
# custom regexes for catching signatures that talon was not able to capture
signature_res = [re.compile(r'(^|\n)-* *(best|all the best|best wishes|sincerely|thank you|thanks|thx|many thanks|thanks again|thanks for your help|thanks for your time' +
    r'|thank you for your help|thank you for your time|cheers|sincerely|regards|warmly|warm regards|best regards|kind regards),* *(\n|$)'),
re.compile(r"(^|\n) *get outlook for ios *(\n|$)"),
re.compile(r"(^|\n) *sent from my \.+(\n|$)"), # note: the dot here should not have been escaped as we are using it as a meta-character (left as is to preserve the version of code sent to Yva)
re.compile(r"(^|\n) *(hi there, *)?\n*([a-z\. ]* is inviting you to a scheduled [a-z ]*zoom meeting|you are invited to a zoom meeting now|join from pc, mac, linux, ios or android)"),
re.compile(r"(^|\n) *if you have any questions regarding our pandemic response plan or need additional support please contact us at( crisis_response@collabera\.com)? or call [\d -]+\. *(\n|$)"),
re.compile(r"(^|\n) *begin forwarded message *: *(\n|$)"),
re.compile(r"(^|\n)(caution *: *-* *external mail *\.*\n)?[\d\w\.\- ]+ actively protects your mailbox by quarantining spam and other unwanted email \.(\n|$)")]
isword_re = re.compile(r"^[a-z0-9]+-?[a-z0-9]*$", re.UNICODE)

words2categories = {}
prefixes2categories = {}

def read_liwc_dictionary(liwc_fn):
    with open(liwc_fn) as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        for row in csvreader:
            for cat, term in zip(header, row):
                term = term.lower().strip()
                if not term:
                    continue
                global prefixes2categories, word2categories
                if ".*" in term:
                    # This is a prefix
                    prefix = term.replace('.*', '')
                    prefix2 = stemmer.stem(prefix)
                    prefixes2categories.setdefault(prefix2, []).append(cat)
                else:
                    # Full word
                    words2categories.setdefault(term, []).append(cat)
    return

def check_im(body):
    """
    Check whether the current email body matches an instant messaging format
    Parameters
    ----------
    body : str
        Raw email body, case retained and tabs replaced
    """
    match = re.search(r'(\n|^)[\w\. ]+\n* *\d{1,2} *: *\d{1,2} +(A|P)M *:.*(\n|$)', body)
    if match:
        return True
    return False

def strip_signature(body):
    """
    Strips signature from an email body. Assumes body has already been converted to lower case.
    Uses both pre-made functions (talon) and custom regexes based on common email closings.
    Parameters
    ----------
    body : str
        Email body to remove signature from
    Returns
    -------
    body : str
        Email body from which signature has already been removed, if found
    """
    # (body, signature) returned when signature found; (body, None) returned when no signature found.
    # from: func extract_signature in talon.signature.bruteforce.py
    body, _ = extract_signature(body)

    for r in signature_res:
        match_obj = re.search(r, body)
        if match_obj:
            body = body[:match_obj.start()]
    return body

def sent_to_toks(sent):
    """
    Converts a sentence to a list of tokens in this sentence.
    Parameters
    ----------
    sent : str
        A sentence to be tokenized
    Returns
    -------
    toks : list
        A list of word tokens
    """
    if len(sent) == 0:
        return []
    if "'" in sent or "’" in sent:
        sent = contractions.fix(sent).lower()
        sent = sent.replace("’", ' ').replace("'", ' ') 
    toks = tokenizer.tokenize(sent)
    return [t for t in toks if isword_re.search(t)]

def cheap_hash(to_hash, n=8):
    """
    Computes a shortened version of the hash of an input string. Hashes all numerical characters into the same key "NUM".
    Reduces storage space; defense against lookup tables
    Collision risk checked - no collision found using 20,000 most common English words
    Parameters
    ----------
    to_hash : str
        A string to be hashed
    Returns
    -------
    hashed : str
        A string that represents the hash of to_hash with length n
    """
    if re.match(r'\d+', to_hash):
        hashed = "NUM"
    else:
        hashed = md5(to_hash.encode('utf-8')).hexdigest()[:n] 
    return hashed

def hash_text(to_hash):
    """
    Hashes a list of tokens using cheap_hash defined in this file
    Parameters
    ----------
    to_hash : list
        A list of tokens of type str
    Returns
    -------
    (to_hash, hashed) : tuple
        A tuple of lists of tokens
    """
    hashed = [cheap_hash(x) for x in to_hash]
    return (to_hash, hashed)

def get_categories_from_word(w):
    """
    Return relevant LIWC categories for a single word
    Parameters
    ----------
    w : str
    Returns
    -------
    cats : list
        A list of categories that the current word belongs to
    """
    cats = []
    if w in words2categories:
        cats += words2categories[w]
    # Check if stem is in prefixes
    pref = stemmer.stem(w)
    if pref in prefixes2categories:
        cats += prefixes2categories[pref]
    cats = list(set(cats))
    return cats

def word_to_liwc_cats(words):
    """
    Converts words to LIWC categories
    Parameters
    ----------
    words : list
        A list of words
    Returns
    -------
    cats : list
        A list of LIWC categories, in the same order as the original words
    """
    cats = [c for w in words for c in get_categories_from_word(w)]
    return cats

def liwc_cats_to_dict(cats):
    """
    Converts a list of LIWC terms to a dict of {term: count} mappings.
    Parameters
    ----------
    cats : list
        A list of words that have already been converted to their respective LIWC categories
    Returns
    -------
    dict
        A dictionary mapping LIWC categories to counts, sorted in freqeuncy of category, from most frequent to least frequent
    """
    countdict = Counter(cats)
    return dict(sorted(countdict.items(), key=itemgetter(1), reverse=True))

def body_to_hash(body, return_liwc=False):
    """
    Hashes email body at the token-level, ignoring case and retaining new lines and sentence structure. 
    Runs on several assumptions that should be met
    1) Assumes one-to-one mapping between word and hash - the same word always maps to the same hash, and vice versa, for the most part.
        Exceptions: all numerical digits [0-9]+ hashed to the string NUM
                    all sentence ending punctuation (.!?) hashed to the string SENT_END
    2) All text is writeable and encoded using unicode
    3) All text in previous emails that may occur in the email bodies of replies or forwarded email are removed
    4) All attachments are removed
    5) All html tags are removed

    Multiple new lines are converted to a single new line. Hashed body always ends in new line, even if body does not end in new line.
    Multiple sentence delimiters - delimiters with spaces in between - are retained as is.
    No sentence delimiter added at the end of lines if no sentence delimiters were present.
    Parameters
    ----------
    body : str
        A string that represents raw email body
    return_liwc (optional) : bool
        A boolean indicating whether LIWC categories should be returned for body. If set to True, assumes that the LIWC dictionary
        file has already been read into two global dictionaries: prefixes2categories and words2categories
    Returns
    -------
    (hashed_body, word2hash, lang, liwc_cats)
        A tuple containing the hashed email body, a dictionary matching words to their hashes, the detected language of email body,
        and optional dictionary matching liwc categories to counts
        lang is a tuple whose first element is detected language and second element is its respective probability
        If body is empty, returns a tuple of empty string and None
    """
    if body is None or len(body.strip()) == 0:
        return ('', {}, ('', None), {}) if return_liwc else ('', {}, ('', None))
    
    body = re.sub(' +', ' ', body.replace('\t', ' ')).strip()
    if check_im(body):
        return ('', {}, ('', None), {}) if return_liwc else ('', {}, ('', None))

    # matches: US, USA, U.S, U.S., U.S.A, U.S.A., u.s, u.s., u.s.a, u.s.a.
    # note: should add word boundaries around regex so that words like USB or USAA are not converted to united states (left as is to preserve the version of code sent to Yva)
    # does not matter much for our purposes given all we care about is that they are not converted to "us", but still important to keep correct
    body = re.sub(r'(USA?|U\.S\.?A?\.?|u\.s\.?a?\.?)', 'united states', body)
    # separate camel cases into distinct words by inserting a space before a capital letter in a word:
    # e.g., FinancialAcumen -> Financial Acumen
    # this is necessary as some email bodies have their new line characters already removed in pre-processing, 
    # making it difficult to tokenize
    body = re.sub(r"([a-z]{2,})([A-Z])", r"\1 \2", body)
    body = body.lower()
    body = strip_signature(body)

    body_lines = re.split('\n+', body)
    hashed_lines = []
    # for LIWC purposes
    cleaned_toks = []
    word2hash = {}
    for line in body_lines:
        if len(line.strip()) > 0:
            sents = re.split(r'[\.!\?]+', line)
            # may contain empty lists
            sents_toks = [sent_to_toks(sent.strip()) for sent in sents]
            hashed_sents = []
            for sent_toks in sents_toks:
                (to_hash, hashed) = hash_text(sent_toks)
                word2hash.update(dict(zip(to_hash, hashed)))
                hashed_sents.append(' '.join(hashed))
                cleaned_toks.extend(to_hash)
            hashed_lines.append(' SENT_END '.join(hashed_sents).strip())
            
    hashed_body = '\n'.join(hashed_lines)
    if len(hashed_body) > 0: hashed_body += '\n'

    if len(cleaned_toks) == 0:
        lang = ('', None)
    else:
        r = model.predict(' '.join(cleaned_toks))
        lang = (r[0][0], r[1][0])

    if return_liwc:
        liwc_cat_counts = liwc_cats_to_dict(word_to_liwc_cats(cleaned_toks))
        return (hashed_body, word2hash, lang, liwc_cat_counts)
    return(hashed_body, word2hash, lang)
