'''
This file even more simplifies quickly importing and exporting python data structures (dicts) to json files
for backup and retrieval purposes.
'''

import json

def export_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def import_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data
