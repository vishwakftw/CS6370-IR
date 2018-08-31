import os
import re
import sys
import requests

from argparse import ArgumentParser
from bs4 import BeautifulSoup


def get_parse_save_book(args):
    book_url = args.book_url
    output_base_path = args.save_path
    if args.verbose:
        print("Book URL: " + book_url)
        print("Save path: " + output_base_path)
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    try:
        r = requests.get(book_url)
    except requests.exceptions.RequestException as e:
        print("There has been a network issue, which has caused an exception. Details below:\n{}".format(e))
        sys.exit(1)

    html_file = r.text

    story_soup = BeautifulSoup(html_file, 'lxml')
    story_text = story_soup.get_text()

    chapter_indices = []
    current_chapter = 1

    while True:
        chapter_regex = 'Chapter ' + str(current_chapter) + '(?![0-9])'
        it = re.finditer(chapter_regex, story_text)
        chapter_ids = [(i.start(), i.end()) for i in it]
        if len(chapter_ids) > 0:
            chapter_index = chapter_ids[-1]
        else:
            break
        chapter_indices.append(chapter_index)
        current_chapter += 1

    # Text that marks the end of the book. Books in Project Gutenberg seem to
    # all have some variation of this, but this exact text will not work for
    # all cases. Written specifically for 'Pride and Prejudice' by Jane
    # Austen. Edit this as applicable for other books.
    end_text = 'End of the Project Gutenberg'
    it = re.finditer(end_text, story_text)
    end_ids = [(i.start(), i.end()) for i in it]
    if len(end_ids) > 0:
        chapter_indices.append(end_ids[-1])
    else:
        chapter_indices.append((len(story_text), len(story_text)))

    stripped_story_text = ''

    for chapter in range(len(chapter_indices) - 1):
        chapter_text = story_text[chapter_indices[chapter][1]: chapter_indices[chapter + 1][0]]
        chapter_text = chapter_text.replace("“", '"').replace("”", '"')  # "curly" is technically not valid
        chapter_output_path = os.path.join(output_base_path,
                                           str(chapter + 1) + '_' + book_url.replace('/', '#') + '_text.txt')
        with open(chapter_output_path, 'w') as f:
            f.write(chapter_text)
        if args.verbose:
            print("Written Chapter " + str(chapter + 1) +
                  " to " + chapter_output_path)
        stripped_story_text += (chapter_text + '\n')

if __name__ == "__main__":
    parser = ArgumentParser(
        description='Given a URL of a book in HTML format, this script fetches \
        the book, splits text into separate chapters, and saves the complete book \
        and each individual chapter in text files. Currently works for "Pride \
        and Prejudice" by Jane Austen from Project Gutenberg.')
    parser.add_argument(
        '--book_url', default='https://www.gutenberg.org/files/1342/1342-h/1342-h.htm', type=str, help='URL of HTML book')
    parser.add_argument('--save_path', default=os.path.join('..', 'data'), type=str,
                        help='Path to the directory where generated text files are to be saved.')
    parser.add_argument('-v', '--verbose', help='Run in verbose mode',
                        dest='verbose', action='store_true')
    args = parser.parse_args()
    get_parse_save_book(args)
