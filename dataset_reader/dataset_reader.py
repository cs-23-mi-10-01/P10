
import os
import csv
import copy

class DatasetReader:
    def __init__(self, params):
        self.params = params
        self.base_directory = params.base_directory
        self.dataset = params.dataset

        self._rows = []
    
    def read_full_dataset(self):
        self._rows = []

        if self.dataset in ["wikidata12k"]:
            full_filename = "triple2id.txt"
            id2entity = {}
            id2relation = {}

            entity2id_path = os.path.join(self.base_directory, "datasets", self.dataset, "entity2id.txt")
            with open(entity2id_path, encoding='utf-8') as identifiers:
                records = csv.DictReader(identifiers, delimiter='\t')
                for row in records:
                    id2entity[row["id"]] = row["entity"]

            relation2id_path = os.path.join(self.base_directory, "datasets", self.dataset, "relation2id.txt")
            with open(relation2id_path, encoding='utf-8') as identifiers:
                records = csv.DictReader(identifiers, delimiter='\t')
                for row in records:
                    id2relation[row["id"]] = row["relation"]

        else:
            full_filename = "full.txt"

        dataset_path = os.path.join(self.base_directory, "datasets", self.dataset, full_filename)
        with open(dataset_path, encoding='utf-8') as full_dataset:
            records = csv.DictReader(full_dataset, delimiter='\t')
            for row in records:
                modified_row = copy.copy(row)

                if self.dataset in ["wikidata12k"]:
                    modified_row["head"] = id2entity[row["head"]]
                    modified_row["relation"] = id2relation[row["relation"]]
                    modified_row["tail"] = id2entity[row["tail"]]
                    if row["time_from"] == "####":
                        modified_row["time_from"] = "-"
                    if row["time_to"] == "####":
                        modified_row["time_to"] = "-"

                self._rows.append(modified_row)
        
    def rows(self):
        return self._rows

