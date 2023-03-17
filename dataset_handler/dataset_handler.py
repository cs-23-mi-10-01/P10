
import os
import csv
import copy

class DatasetHandler:
    def __init__(self, params):
        self.params = params
        self.base_directory = params.base_directory
        self.dataset = params.dataset

        self._rows = []
        self._id2entity = {}
        self._id2relation = {}

        if self.dataset in ["wikidata12k", "yago11k"]:
            entity2id_path = os.path.join(self.base_directory, "datasets", self.dataset, "entity2id.txt")
            with open(entity2id_path, encoding='utf-8') as identifiers:
                records = csv.DictReader(identifiers, fieldnames=['entity', 'id'], delimiter='\t')
                for row in records:
                    self._id2entity[row["id"]] = row["entity"]

            relation2id_path = os.path.join(self.base_directory, "datasets", self.dataset, "relation2id.txt")
            with open(relation2id_path, encoding='utf-8') as identifiers:
                records = csv.DictReader(identifiers, fieldnames=['relation', 'id', 'relation_name', 'types'], delimiter='\t')
                for row in records:
                    self._id2relation[row["id"]] = row["relation"]
    
    def read_full_dataset(self):
        self._rows = []

        if self.dataset in ["wikidata12k", "yago11k"]:
            full_filename = "triple2id.txt"
        else:
            full_filename = "full.txt"

        dataset_path = os.path.join(self.base_directory, "datasets", self.dataset, full_filename)

        self._read_file(dataset_path)
        
        self._rows.sort(key=lambda row: row["head"]+";"+row["relation"]+";"+row["tail"]+";"+row["start_timestamp"]+";"+row["end_timestamp"], reverse=False)

    def find_in_rows(self, head="*", relation="*", tail="*", start_timestamp="*", end_timestamp="*"):
        ret_rows = []

        for row in self._rows:
            if head == "*" or row["head"] == head:
                if relation == "*" or row["relation"] == relation:
                    if tail == "*" or row["tail"] == tail:
                        if start_timestamp == "*" or row["start_timestamp"] == start_timestamp:
                            if end_timestamp == "*" or row["end_timestamp"] == end_timestamp:
                                ret_rows.append(row)
            elif head < row["head"]:
                break
        
        return ret_rows

    def read_original_splits(self):
        self._rows = []

        original_path = os.path.join(self.base_directory, "datasets", self.dataset, "original")
        test_path = os.path.join(original_path, "test.txt")
        train_path = os.path.join(original_path, "train.txt")
        valid_path = os.path.join(original_path, "valid.txt")

        for row in self._read_file(test_path):
            row['split'] = 'test'
        for row in self._read_file(train_path):
            row['split'] = 'train'
        for row in self._read_file(valid_path):
            row['split'] = 'valid'

    def _read_file(self, path):
        new_rows = []

        with open(path, encoding='utf-8') as full_dataset:
            if self.dataset in ['icews14']:
                fieldnames=['head', 'relation', 'tail', 'timestamp']
            elif self.dataset in ['wikidata12k', 'yago11k']:
                fieldnames=['head', 'relation', 'tail', 'start_timestamp', 'end_timestamp']
            
            records = csv.DictReader(full_dataset, fieldnames=fieldnames, delimiter='\t')
            for row in records:
                modified_row = copy.copy(row)

                if self.dataset in ["wikidata12k", "yago11k"]:
                    modified_row["head"] = self._id2entity[row["head"]]
                    modified_row["relation"] = self._id2relation[row["relation"]]
                    modified_row["tail"] = self._id2entity[row["tail"]]
                    if row["start_timestamp"] == "####":
                        modified_row["start_timestamp"] = "-"
                    if row["end_timestamp"] == "####":
                        modified_row["end_timestamp"] = "-"

                new_rows.append(modified_row)
                self._rows.append(modified_row)
        
        return new_rows
        
    def rows(self):
        return self._rows

