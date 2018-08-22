import csv
import json

from argparse import ArgumentParser as AP

p = AP()
p.add_argument('--path', type=str, required=True,
                            help='JSON file to be converted to CSV.')
p = p.parse_args()

with open(p.path) as f:
	j = json.load(f)
	c = csv.writer(open("output.csv", "w"))

	c.writerow(["TextId","Text"])

	for key in j:
	    c.writerow([key,j[key]])