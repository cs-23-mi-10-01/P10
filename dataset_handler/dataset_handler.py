
import os
import csv
import copy

class DatasetHandler:
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
                    if row["start_timestamp"] == "####":
                        modified_row["start_timestamp"] = "-"
                    if row["end_timestamp"] == "####":
                        modified_row["end_timestamp"] = "-"

                self._rows.append(modified_row)
        
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
        
    def rows(self):
        return self._rows

