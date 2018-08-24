import os
import string
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import Counter

from nltk import word_tokenize

from argparse import ArgumentParser as AP

from matplotlib import pyplot as plt


p = AP()
p.add_argument('--datasrc', type=str, required=True,
                            help='Source for the documents. To be presented in CSV format')
p.add_argument('--log_scale', action='store_true',
                              help='Toggle to generate graphs in log-scale')
p.add_argument('--linear_fit', action='store_true',
                               help='Toggle to compute the linear fit')
p = p.parse_args()

# bar format for tqdm progress bar
BAR_FORMAT = '{desc}{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt}'

# Use Pandas for storing the text
# Get the textual part of the dataset, and separate out the text and the doc ID
doc_db = pd.read_csv(p.datasrc)
doc_db_text = doc_db['Text']
doc_db_id = doc_db['TextId']

# Iterate over each document and perform the following
# 1. Case folding: bring all documents to lower case
# 2. Process text by tokenizing and removing punctuation and digits
# 3. Collect all the unique vocabulary from all the documents into a Counter object. This sorts by default

TRANSLATION_TABLE = str.maketrans('', '', string.punctuation + string.digits)  # used to remove punctuation and digits

# all_pairs contains the cumulative count of distinct terms and the total tokens seen
all_pairs = [[0, 0]]
all_terms = set()
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
    tmp = [tmp_w for tmp_w in tmp if tmp_w != '']  # Remove empty string results

    all_terms.update(set(tmp))
    all_pairs.append([all_pairs[-1][0] + len(tmp), len(all_terms)])
all_pairs = np.array(all_pairs[1:])
if all_warns != 0:
    warnings.warn("There were {} empty documents".format(all_warns), RuntimeWarning)

# Plot the #tokens vs. #terms graph -> Heaps' Law plot
plt.figure(figsize=(10, 8))
plt.title('# Tokens vs. # Terms for the corpus: {}'.format(os.path.basename(p.datasrc)), fontsize=20)
plt.xlabel('Cumulative number of Tokens', fontsize=15)
plt.ylabel('Cumulative number of Terms', fontsize=15)
if p.log_scale:
    plt.xscale('log')
    plt.yscale('log')
plt.plot(all_pairs[:, 0], all_pairs[:, 1], 'b-', linewidth=3.0, alpha=0.4)
plt.scatter(all_pairs[:, 0], all_pairs[:, 1], 2.0, color='k')
plt.tight_layout()
plt.show()

if p.linear_fit:
    xs = np.log10(all_pairs[:, 0])
    ys = np.log10(all_pairs[:, 1])
    solution = np.polyfit(xs, ys, 1)
    print("slope: {}, intercept: {}".format(round(solution[0], 3), round(solution[1], 3)))
