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
                            help='Source for the documents. To be presented in CSV format')
p.add_argument('--remove_stopword', action='store_true',
                                    help='Toggle to remove stop words from the text')
p.add_argument('--stem', action='store_true',
                         help='Toggle to stem the terms collected from the text')
p.add_argument('--lemmatize', action='store_true',
                              help='Toggle to lemmative the terms collected from the text')
p.add_argument('--log_scale', action='store_true',
                              help='Toggle to generate graphs in log-scale')
p.add_argument('--linear_fit', action='store_true',
                               help='Toggle to compute the linear fit')
p = p.parse_args()

NO_STOPWORD = p.remove_stopword
STEM = p.stem
LEMMATIZE = p.lemmatize

# bar format for tqdm progress bar
BAR_FORMAT = '{desc}{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt}'

# Use Pandas for storing the text
# Get the textual part of the dataset, and separate out the text and the doc ID
doc_db = pd.read_csv(p.datasrc)
doc_db_text = doc_db['Text']
doc_db_id = doc_db['TextId']

# Iterate over each document and perform the following
# 1. Case folding: bring all documents to lower case
# 2. Process text
#   2a. Tokenize the documents, and remove punctuation + digits
#   2b. Remove stopwords (If required)
#   2c. Stemming (If required)
#   2d. Lemmatize (If required)
# 3. Collect all the unique vocabulary from all the documents into a Counter object. This sorts by default

TRANSLATION_TABLE = str.maketrans('', '', string.punctuation + string.digits)  # used to remove punctuation and digits

if NO_STOPWORD:
    ALL_STOPWORDS = set(stopwords.words('english'))  # Set is faster because set uses hashes
if STEM:
    stemmer = PorterStemmer()
if LEMMATIZE:
    lemmatizer = WordNetLemmatizer()

all_terms = []
all_warns = 0
for doc_id in tqdm(range(len(doc_db_id)), ascii=True, desc='processing', bar_format=BAR_FORMAT):
    tmp = doc_db_text[doc_id]
    try:
        tmp = tmp.lower()  # Case folding
    except:
        all_warns += 1
        continue
    tmp = tmp.translate(TRANSLATION_TABLE)  # Removal of punctuation and digits
    tmp = word_tokenize(tmp)  # Tokenization
    if NO_STOPWORD:
        tmp = [tmp_w for tmp_w in tmp if tmp_w not in ALL_STOPWORDS]  # Stopword removal
    if STEM:
        tmp = [stemmer.stem(tmp_w) for tmp_w in tmp]  # Stemming
    if LEMMATIZE:
        tmp = [lemmatizer.lemmatize(tmp_w) for tmp_w in tmp]  # Lemmatize

    tmp = [tmp_w for tmp_w in tmp if tmp_w != '']  # Stemming and Lemmatization can cause empty string results
    all_terms += tmp

all_counter = Counter(all_terms)
if all_warns != 0:
    warnings.warn("There were {} empty documents".format(all_warns), RuntimeWarning)

# Print the top 20 terms
print("Top 20 common terms")
for mcw in all_counter.most_common(20):
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
