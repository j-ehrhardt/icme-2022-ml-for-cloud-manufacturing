'''
This file handles importing inflation data from csv files provided by the Italian National Institute of Statistics.
'''

# Data taken from http://dati.istat.it/Index.aspx?QueryId=24868&lang=en#

import csv

def inflation(csvfile):
    # Import relevant lines from file: 132 month, Jan 2011 - Dec 2021
    org_indexed = list()
    with open(csvfile, encoding='utf-8') as file:
        csvreader = csv.reader(file)
        for line in csvreader:
            if line[5] == 'index number' and line[7] == 'all items':
                org_indexed.append(line)

    # Rebase index to first period (Jan 2011) - data is index with 2015 being 100
    inflation = list()
    inflation_rebase_factor = float(org_indexed[0][10]) / 100
    for line in org_indexed:
        for week in range(1, 5):
            inflation.append((float(line[10]) / inflation_rebase_factor, line[8] + '-' + str(week)))

    return inflation