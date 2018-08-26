import os
import re
import sys
import requests

from bs4 import BeautifulSoup


book_url = 'https://www.gutenberg.org/files/1342/1342-h/1342-h.htm'
output_base_path = os.path.join('..', 'data')  # path to write output txt files

# Can define specific Chapter styles if need more general, for now, using
# Pride and Prejudice

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

end_text = 'End of the Project Gutenberg'  # Not always true
it = re.finditer(end_text, story_text)
end_ids = [(i.start(), i.end()) for i in it]
if len(end_ids) > 0:
    chapter_indices.append(end_ids[-1])
else:
    chapter_indices.append((len(story_text), len(story_text)))

stripped_story_text = ''

for chapter in range(len(chapter_indices) - 1):
    chapter_text = story_text[chapter_indices[chapter][1]: chapter_indices[chapter + 1][0]]
    with open(os.path.join(output_base_path, str(chapter + 1) + '_' + 'Book' + '_text.txt'), 'w') as f:
        f.write(chapter_text)
    stripped_story_text += (chapter_text + '\n')

with open(os.path.join(output_base_path, 'Book' + '_text.txt'), 'w') as f:
    f.write(stripped_story_text)
