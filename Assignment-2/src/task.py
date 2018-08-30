import os
import string
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import Counter

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from argparse import ArgumentParser as AP

from matplotlib import pyplot as plt


p = AP()
p.add_argument('--datasrc', type=str, required=True,
                            help='Directory where the documents are stored. Files to be presented in TXT format')
p.add_argument('--top_k', type=int, default=20,
               help='number of top words for the analysis')
p.add_argument('--top_k_scheme', default='frequency',
               choices=['frequency', 'entropy', 'custom'], 
               help='scheme to use to find top_k words for the analysis')
p.add_argument('--chapter_wise', action='store_true',
               help='Toggle to perform the analysis chapter wise')
p.add_argument('--log_scale', action='store_true',
                              help='Toggle to generate graphs in log-scale')

p = p.parse_args()

analysis_type = "chapter" if p.chapter_wise else "complete book"
top_k = p.top_k
top_k_scheme = p.top_k_scheme

# bar format for tqdm progress bar
BAR_FORMAT = '{desc}{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt}'

# Get all documents from the directory and store their path
doc_db_filepath = []

for root, dirs, files in os.walk(p.datasrc):
    for filename in files:
        doc_db_filepath.append(os.path.join(p.datasrc, filename))

# Iterate over each document and perform the following
# 1. Case folding: bring all documents to lower case
# 2. Process text
#   2a. Tokenize the documents, and remove punctuation + digits
#   2b. Remove stopwords
#   2c. Stemming
#   2d. Lemmatize
# 3. Collect all the unique vocabulary from all the documents into a Counter object. This sorts by default

TRANSLATION_TABLE = str.maketrans('', '', string.punctuation + string.digits)  # used to remove punctuation and digits

ALL_STOPWORDS = set(stopwords.words('english'))  # Set is faster because set uses hashes
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

if analysis_type == "chapter":
    chapter_term_frequency = [[]] * len(doc_db_filepath)
    chapter_counter = []

book_term_frequency = []
all_warns = 0
for doc_id in tqdm(range(len(doc_db_filepath)), ascii=True, desc='processing', bar_format=BAR_FORMAT):
    with open(doc_db_filepath[doc_id], "r") as file:
        tmp = file.read()
        try:
            tmp = tmp.lower()  # Case folding
        except:
            all_warns += 1
            continue
        tmp = tmp.translate(TRANSLATION_TABLE)  # Removal of punctuation and digits
        tmp = word_tokenize(tmp)  # Tokenization
        tmp = [tmp_w for tmp_w in tmp if tmp_w not in ALL_STOPWORDS]  # Stopword removal
        tmp = [stemmer.stem(tmp_w) for tmp_w in tmp]  # Stemming
        tmp = [lemmatizer.lemmatize(tmp_w) for tmp_w in tmp]  # Lemmatize

        tmp = [tmp_w for tmp_w in tmp if tmp_w != '']  # Stemming and Lemmatization can cause empty string results
        if analysis_type == "chapter":
            chapter_term_frequency[doc_id] = chapter_term_frequency[doc_id] + tmp
        
        book_term_frequency += tmp

if all_warns != 0:
    warnings.warn("There were {} empty documents".format(all_warns), RuntimeWarning)

# Print the top k terms
if analysis_type == "chapter":
    for ch in chapter_term_frequency[:10]:
        chapter_counter.append( Counter(ch))

    for cnt in chapter_counter:
        print("Top {} common terms using {} performed {} wise for 10 chapters.".format(
            top_k, p.top_k_scheme, analysis_type))
        for mcw in cnt.most_common(top_k):
            print("{}: {}".format(mcw[0], mcw[1]))
        print("----")

else:
    all_counter = Counter(book_term_frequency)
    for mcw in all_counter.most_common(top_k):
        print("{}: {}".format(mcw[0], mcw[1]))
