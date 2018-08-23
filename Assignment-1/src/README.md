### Assignment 1

#### What is in this folder?
+ There are three Python scripts: `task1.py`, `task2.py` and `json-to-csv.py`.

+ `task1.py` is used for empirically validating **Zipf's Law**, and `task2.py` is used for empirically validating **Heaps' Law**.

+ There were two datasets provided for the purposes of this assignment.
    + One of them was a CSV file, with each row representing a distinct document.
    + The other was a JSON file. Each entry is a unique hash identifier for the document it represents. `json-to-csv.py` converts this JSON file into CSV format, analogous to the first dataset.

#### How do I run the scripts?
+ You have to pass command line arguments to the scripts, which are self-explanatory. To find out what each script requires, run:
```
python <script> -h
```
+ Arguments enclosed in square brackets are optional.

#### Additional packages required
+ `tqdm` (for the nice-looking progress bar)
+ `nltk` (for the parsing of the text)
+ `matplotlib` (for plotting purposes)
+ `pandas` (for storing the dataset efficiently)
+ `numpy` (for arrays, and advanced indexing)

+ If you are running a Anaconda / Miniconda environment, run `conda install --file requirements.txt` to install these packages.
+ If you are using `pip` as your package manager, run `pip install --user -r requirements.txt` to install these packages.

+ `nltk` requires a downloaded corpus for the stopword removal. If there are issues with `nltk` after installing regarding `nltk_data`, please run `python -c "import nltk ; nltk.download('popular')"`.

+ All these scripts are run using Python 3 only. Please don't except the scripts to run in Python 2.
