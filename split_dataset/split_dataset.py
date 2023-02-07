
import os
import csv
import copy
import random

from pathlib import Path

class SplitDataset:
    def __init__(self, params):
        self.params = params
        self.base_directory = params.base_directory
        self.dataset = params.dataset
        self.valid_percentage = 0.15
        self.test_percentage = 0.15

        random.seed(100)
        
        self.rows = []
        self.entitiy_count = {}
        self.relation_count = {}
        self.timestamp_count = {}

    def split(self):
        dataset_path = os.path.join(self.base_directory, "datasets", self.dataset, "full.txt")

        self.rows = []
        with open(dataset_path, encoding='utf-8') as full_dataset:
            records = csv.DictReader(full_dataset, delimiter='\t')
            for row in records:
                self.rows.append(row)
                self._add_element(row['head'], self.entitiy_count)
                self._add_element(row['relation'], self.relation_count)
                self._add_element(row['tail'], self.entitiy_count)
                self._add_element(row['timestamp'], self.timestamp_count)

        for i in ["1"]:
            self._split_once(i)

    def _add_element(self, element, element_dir):
        if element not in element_dir.keys():
            element_dir[element] = 1
        else:
            element_dir[element] += 1
    
    def _subtract_element(self, element, element_dir):
        element_dir[element] -= 1

    def _split_once(self, name):
        entity_count = copy.copy(self.entitiy_count)
        relation_count = copy.copy(self.relation_count)
        timestamp_count = copy.copy(self.timestamp_count)
        rows = copy.copy(self.rows)
        random.shuffle(rows)

        num_valid = len(rows) * self.valid_percentage
        num_test = len(rows) * self.test_percentage

        for row in rows:
            row['split'] = 'train'

        for row in rows:
            if num_valid == 0 and num_test == 0:
                break

            if entity_count[row['head']] > 2 and \
               relation_count[row['relation']] > 1 and \
               entity_count[row['tail']] > 2 and \
               timestamp_count[row['timestamp']] > 1:
                self._subtract_element(row['head'], entity_count)
                self._subtract_element(row['relation'], relation_count)
                self._subtract_element(row['tail'], entity_count)
                self._subtract_element(row['timestamp'], timestamp_count)

                if num_valid > 0:
                    row['split'] = 'valid'
                    num_valid -= 1
                    continue
                if num_test > 0:
                    row['split'] = 'test'
                    num_test -= 1
                    continue
        
        self._write_csv(rows, name, 'train')
        self._write_csv(rows, name, 'valid')
        self._write_csv(rows, name, 'test')

    def _write_csv(self, rows, name, split):
        text = ""
        for row in [row for row in rows if row['split'] is split]:
            text = text + row['head'] + "\t" + row['relation'] + "\t" + row['tail'] + "\t" + row['timestamp'] + "\n"
        path = os.path.join(self.base_directory, "datasets", self.dataset, name, split + ".txt")
        self._write(path, text)

    def _write(self, path, text):
        Path(path).touch(exist_ok=True)
        out_file = open(path, "w", encoding="utf8")
        out_file.write(text)
        out_file.close()

            

