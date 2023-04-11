
import os
import json
from pathlib import Path
import copy

def read_json(path):
    print("Reading from file " + path + "...")
    in_file = open(path, "r", encoding="utf8")
    dict = json.load(in_file)
    in_file.close()
    return dict

def write_json(path, dict):
    print("Writing to file " + path + "...")
    touch(path)
    out_file = open(path, "w", encoding="utf8")
    json.dump(dict, out_file, indent=4)
    out_file.close()

def simulate_dates(start, end, delta):
    dates = []
    sim_date = copy.copy(start)
    while sim_date != end:
        dates.append(copy.copy(sim_date))
        sim_date = sim_date + delta
    return dates + [copy.copy(end)]


def remove_unwanted_symbols_from_str(str):
    return str.replace(' ', ' ')

def remove_unwanted_symbols(dict): 
    while True:
        target_key = None
        for key in dict.keys():
            if ' ' in key:
                target_key = key
                break
        if target_key is not None:
            dict[remove_unwanted_symbols_from_str(target_key)] = dict.pop(target_key)
        else:
            break

def touch(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    Path(path).touch(exist_ok=True)

def write(path, text):
    touch(path)
    out_file = open(path, "w", encoding="utf8")
    out_file.write(text)
    out_file.close()

def year_to_iso_format(year):
    modified_year = year
    if modified_year == '-' or modified_year == '####':
        modified_year = "0001"
    while len(modified_year) < 4:
        modified_year = "0" + modified_year
    return modified_year + "-01-01"

def exists(path):
    return os.path.exists(path)