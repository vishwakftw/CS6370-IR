import os
import string
import warnings
import numpy as np

from tqdm import tqdm
from collections import Counter

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from argparse import ArgumentParser as AP

from matplotlib import pyplot as plt


def compute_entropy_from_matrix(matrix):
    """
    Utility function to compute the entropy from a matrix
    """
    count_list = np.sum(matrix, axis=0)
    prob_matrix = matrix / count_list
    prob_matrix[np.isnan(prob_matrix)] = 0  # This will make 0/0 errors not cause issue later on
    prob_matrix = prob_matrix * np.log(prob_matrix)
    prob_matrix[np.isinf(prob_matrix)] = 0  # This will make log(0) = -inf not cause issues later on
    clf_list = -np.sum(prob_matrix, axis=0)
    return clf_list


p = AP()
p.add_argument('--datasrc', type=str, required=True,
                            help='Directory where the documents are stored. Files to be presented in TXT format')
p.add_argument('--top_k', type=int, default=20,
               help='Number of top words for the analysis')
p.add_argument('--top_k_scheme', default='frequency',
               choices=['frequency', 'entropy', 'custom'], 
               help='Scheme to use to find top_k words for the analysis')
p.add_argument('--chapter_wise', action='store_true',
               help='Toggle to perform the analysis chapter wise')
p.add_argument('--scatter_freq_entropy', action='store_true',
                                         help='Toggle to get the scatter plot of entropy and frequency')

p = p.parse_args()

if p.chapter_wise and (p.top_k_scheme != "frequency" or p.scatter_freq_entropy):
    raise ValueError("Chapter-wise top-k words can be only found using a frequency-based technique")

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
#   2c. Lemmatize
# 3. Collect all the unique vocabulary from all the documents into a Counter object. This sorts by default

TRANSLATION_TABLE = str.maketrans('', '', string.punctuation + string.digits)  # used to remove punctuation and digits

ALL_STOPWORDS = set(stopwords.words('english'))  # Set is faster because set uses hashes
lemmatizer = WordNetLemmatizer()

# The process is as below
# 1. We go chapter-by-chapter
# 2. Process the text (stopword removal and lemmatization)
# 3. Memoize a Counter object for each chapter : this is a dictionary of form {"word": # of instances}
# 4. Memoize a set of words which have occurred as well
# This means that we now have all the counts for every word occurring in the book for every chapter

chapter_counters = []
cur_dictionary = set()

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
        tmp = [lemmatizer.lemmatize(tmp_w) for tmp_w in tmp]  # Lemmatize
        tmp = [tmp_w for tmp_w in tmp if tmp_w != '']  # Stemming and Lemmatization can cause empty string results

        chapter_counters.append((doc_id, Counter(tmp)))
        cur_dictionary.update(tmp)

cur_dictionary = list(cur_dictionary)

if all_warns != 0:
    warnings.warn("There were {} empty documents".format(all_warns), RuntimeWarning)

# Now we have obtained the required information
# Let's proceed to get the top-k words
if p.chapter_wise:
    # This is top-k words chapter wise
    chapter_counters = sorted(chapter_counters, key=lambda x: sum(x[1].values()), reverse=True)

    # If the number of chapter is greater than 10, then we consider the first 10 chapters with
    # the largest number of terms
    if len(doc_db_filepath) > 10:
        chapter_counters = chapter_counters[:10]

    for doc_id, cnt in chapter_counters:
        # First print the top-k words and then generate the distribution plot
        print("Top-{} words in chapter {}".format(p.top_k, doc_id))
        for mcw in cnt.most_common(p.top_k):
            print("{}: {}".format(mcw[0], mcw[1]))
        print("=====\n")

        plt.figure(figsize=(10, 8))
        plt.title('Frequency distribution for chapter {}'.format(doc_id))
        plt.hist(cnt.values(), bins='auto')
        plt.xlabel('Number of occurrences')
        plt.ylabel('Number of terms')
        plt.show()

else:
    # This is top-k for the entire book
    # Scheme matters here
    # We will build a matrix of counts per chapters for every term in the dictionary
    # Every row in the dictionary is the count for a different terms in one chapter
    term_matrix = np.array([[cnt[t] for t in cur_dictionary] for cnt in chapter_counters])

    if p.top_k_scheme == "frequency":
        term_clf_list = np.sum(term_matrix, axis=0)
        indexes = np.argsort(term_clf_list)[::-1]  # Reverse order i.e., descending

    if p.top_k_scheme == "entropy":
        term_clf_list = compute_entropy_from_matrix(term_matrix)
        indexes = np.argsort(term_clf_list)[::-1]  # Reverse order i.e., descending

    if p.top_k_scheme == "custom":
        # TODO

    print("Top-{} words in the book by {}".format(p.top_k, p.top_k_scheme))
    for index in indexes[:p.top_k]:
        print("{}: {}".format(cur_dictionary[index], term_clf_list[index]))

    plt.figure(figsize=(10, 8))
    plt.title('Frequency distribution for book')
    plt.hist(term_clf_list, bins='auto')
    plt.show()

    if p.scatter_freq_entropy:
        # Compute the total occurrences of every term in the book
        term_count_list = np.sum(term_matrix, axis=0)
        term_entropy_list = compute_entropy_from_matrix(term_matrix)
        plt.figure(figsize=(10, 8))
        plt.title('Entropy vs. Frequency plot')
        plt.xlabel('Entropy')
        plt.ylabel('Frequency')
        plt.scatter(term_entropy_list, term_count_list, 3.0, color='b', alpha=0.7)
        plt.show()
