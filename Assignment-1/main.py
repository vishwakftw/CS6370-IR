from nltk import wordpunct_tokenize
from argparse import ArgumentParser as AP

p = AP()
p.add_argument('--datasrc', type=str, required=True,
                            help='Source for the text')
p.add_argument('--remove_stopword', action='store_true',
                                    help='Toggle to remove stop words from the text')
p.add_argument('--stem', action='store_true',
                         help='Toggle to stem the terms collected from the text')
p.add_argument('--lemmatize', action='store_true',
                              help='Toggle to lemmative the terms collected from the text')
p = p.parse_args()

# Get raw text as one big string instance
cur_text = open(p.datasrc, 'r').read()

# It is important to do case-folding, otherwise
# "Hello" and "hello" will be considered to be different
cur_text = cur_text.lower()

# Now tokenize the string using wordpunct_tokenize
tokens = wordpunct_tokenize(cur_text)
