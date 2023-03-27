
import os
from pathlib import Path

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