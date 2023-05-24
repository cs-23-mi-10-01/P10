
import os
import json
from pathlib import Path
from typing import List
import copy

######################################################      READ/WRITE       ########################################################
def touch(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    Path(path).touch(exist_ok=True)

def exists(path):
    return os.path.exists(path)

def read_text(path):
        in_file = open(path, "r", encoding="utf8")
        text = in_file.read()
        in_file.close()
        return text

def write(path, text):
    touch(path)
    out_file = open(path, "w", encoding="utf8")
    out_file.write(text)
    out_file.close()

def read_json(path, write=True):
    if write: print("Reading from file " + path + "...")
    in_file = open(path, "r", encoding="utf8")
    dict = json.load(in_file)
    in_file.close()
    return dict

def write_json(path, dict, write=True):
    if write: print("Writing to file " + path + "...")
    touch(path)
    out_file = open(path, "w", encoding="utf8")
    json.dump(dict, out_file, indent=4)
    out_file.close()

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

######################################################      DATE ARITHMETIC       ########################################################

def simulate_dates(start, end, delta):
    dates = []
    sim_date = copy.copy(start)
    while sim_date != end:
        dates.append(copy.copy(sim_date))
        sim_date = sim_date + delta
    return dates + [copy.copy(end)]

def year_to_iso_format(year):
    modified_year = year
    if modified_year == '-' or modified_year == '####':
        modified_year = "0001"
    while len(modified_year) < 4:
        modified_year = "0" + modified_year
    return modified_year + "-01-01"

def date_to_iso(date):
    return f"{date.year:04d}-{date.month:02d}-{date.day:02d}"


######################################################      DICTIONARY ACCESS       ########################################################

# adapted from https://stackoverflow.com/questions/39818669/dynamically-accessing-nested-dictionary-keys
def getdctval(dct, keys: List):
    data = dct
    for k in keys:
        data = data[k]
    return data

# also adapted
def setval(dct, keys: List, val) -> None:
        data = dct
        lastkey = keys[-1]
        for k in keys[:-1]:  # when assigning drill down to *second* last key
            data = data[k]
        data[lastkey] = val

######################################################      FIGURE HELPER FUNCTIONS       ########################################################

def divide_into_buckets(coordinates, buckets=-1):
    first_time_float = float(coordinates[0][0])
    last_time_float = float(coordinates[-1][0])

    if buckets != -1:
        interval = (last_time_float - first_time_float) / buckets
        for i in range(buckets + 1):
            coordinate_indexes_in_bucket = []
            for j in range(len(coordinates)):
                if coordinates[j][0] >= first_time_float + interval*i and \
                    coordinates[j][0] < first_time_float + interval*(i+1):
                    coordinate_indexes_in_bucket.append(j)

            sum_of_vals_in_bucket = 0
            total_date_intervals_in_bucket = 0.0
            for index in coordinate_indexes_in_bucket:
                total_date_intervals_in_bucket += float(coordinates[index][0])
                sum_of_vals_in_bucket += float(coordinates[index][1])
            average_date_interval = total_date_intervals_in_bucket / len(coordinate_indexes_in_bucket)
            average_no_of_facts = sum_of_vals_in_bucket / len(coordinate_indexes_in_bucket)
            average_coord = [average_date_interval, average_no_of_facts]

            for j in range(len(coordinate_indexes_in_bucket) -1, 0, -1):
                coordinates.pop(coordinate_indexes_in_bucket[j])
            coordinates[coordinate_indexes_in_bucket[0]] = average_coord
    
    return coordinates