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
p.add_argument('--top_k', type=int, required=True,
               help='How many words to consider for analysis')
p.add_argument('--log_scale', action='store_true',
                              help='Toggle to generate graphs in log-scale')
p.add_argument('--linear_fit', action='store_true',
                               help='Toggle to compute the linear fit')
p = p.parse_args()

top_k = p.top_k
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

all_terms = []
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
        all_terms += tmp

all_counter = Counter(all_terms)
if all_warns != 0:
    warnings.warn("There were {} empty documents".format(all_warns), RuntimeWarning)

# Print the top k terms
print("Top {} common terms using frequency".format(top_k))
for mcw in all_counter.most_common(top_k):
    print("{}: {}".format(mcw[0], mcw[1]))

# Plot the frequency vs. rank graph
plt.figure(figsize=(10, 8))
plt.title('Frequency vs. Rank for the corpus: {}'.format(os.path.basename(p.datasrc)), fontsize=20)
plt.xlabel('Rank (1 to {})'.format(len(all_counter)), fontsize=15)
plt.ylabel('Frequency', fontsize=15)
if p.log_scale:
    plt.xscale('log')
    plt.yscale('log')
plt.plot(list(range(1, len(all_counter) + 1)), sorted(all_counter.values(), reverse=True),
         'b-', linewidth=3.0, alpha=0.4)
plt.scatter(list(range(1, len(all_counter) + 1)), sorted(all_counter.values(), reverse=True),
            2.0, color='k')
plt.tight_layout()
plt.show()

if p.linear_fit:
    xs = np.log10(np.arange(1, len(all_counter) + 1, 1))
    ys = np.log10(np.array(sorted(all_counter.values(), reverse=True)))
    solution = np.polyfit(xs, ys, 1)
    print("negative slope: {}, intercept: {}".format(round(-solution[0], 3), round(solution[1], 3)))
