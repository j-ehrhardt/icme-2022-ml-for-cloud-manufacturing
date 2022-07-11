'''
This file is used as an easy to use importer for data on italian municipalities population
provided by the Italian National Institute of Statistics. List is filtered and returned for
the 100 largest municipalities.
'''

# Data taken from: https://demo.istat.it/popres/download.php?anno=2022&lingua=eng
# Compressed csv file can be downloaded at: https://demo.istat.it/pop2022/dati/comuni.zip

import csv
import json

# Read cities from csv, parse data and sort
def top100(csvfile):
    municipalities = dict()
    with open(csvfile, encoding='utf-8') as file:
        csvreader = csv.reader(file)
        for line in csvreader:
            if len(line) > 1 and line[2] == '999':
                municipalities[line[1]] = int(line[10]) + int(line[-1])

    return sorted(municipalities, key=municipalities.get, reverse= True)[:100]

# Read distances from json
def distances(file_locator):
    with open(file_locator) as json_file:
        data = json.load(json_file)
    return data