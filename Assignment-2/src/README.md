### Assignment 2

#### What is in this folder?
+ There are two Python scripts: `parse_book.py` and `task.py`.

+ `parse_book.py` obtains the book from a given URL and separates it into chapters as specified in the assignment statement. Please note that `/` is not a valid character in a filename, which is why the `/`s in the URL have been replaced with `#`s.

+ `task.py` performs the required tasks on the book saved.

#### How do I run the scripts?
+ You have to pass command line arguments to the scripts, which are self-explanatory. To find out what each script requires, run:
```
python <script> -h
```
+ Arguments enclosed in square brackets are optional.

#### Additional packages required
+ `bs4` (for parsing the HTML source to get the raw text version of the book) and `lxml` for a faster parser
+ `tqdm` (for the nice-looking progress bar)
+ `nltk` (for the parsing of the text)
+ `matplotlib` (for plotting purposes)
+ `numpy` (for arrays, and advanced indexing)

+ If you are running a Anaconda / Miniconda environment, run `conda install --file requirements.txt` to install these packages.
+ If you are using `pip` as your package manager, run `pip install --user -r requirements.txt` to install these packages.

+ `nltk` requires a downloaded corpus for the stopword removal. If there are issues with `nltk` after installing regarding `nltk_data`, please run `python -c "import nltk ; nltk.download('popular')"`.

+ All these scripts are run using Python 3 only. Please don't expect the scripts to run in Python 2.
