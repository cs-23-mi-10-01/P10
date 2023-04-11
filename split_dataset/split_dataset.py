
import os
import csv
import copy
import random
import re

from scripts import write
from dataset_handler.dataset_handler import DatasetHandler

class SplitDataset:
    def __init__(self, params):
        self.params = params
        self.base_directory = params.base_directory
        self.dataset = params.dataset
        self.valid_percentage = 0.15
        self.test_percentage = 0.15
        
        self.rows = []
        self.entitiy_count = {}
        self.relation_count = {}
        self.timestamp_count = {}

    def split(self):
        self.params.timer.start("split " + self.dataset)
        random.seed(100)
        print("Pre-processing dataset " + self.dataset + " for train/valid/test sets...")

        reader = DatasetHandler(self.params)
        reader.read_full_dataset()
        
        if self.dataset in ['wikidata11k']:
            for row in reader.rows():
                row['handled'] = False

        processed_rows = []
        i = 0
        for row in reader.rows():
            if i % 1000 == 0:
                print("Pre-processing rows " + str(i) + "-" + str(i + 999) + " out of " + str(len(reader.rows())) + " rows.")
            i += 1

            if not self._include_row(row):
                continue

            if self.dataset in ['wikidata11k']:
                if row['handled']:
                    continue

            self._add_element(row['head'], self.entitiy_count)
            self._add_element(row['relation'], self.relation_count)
            self._add_element(row['tail'], self.entitiy_count)

            if self.dataset in ['icews14']:
                self._add_element(row['timestamp'], self.timestamp_count)

            if self.dataset in ['wikidata11k']:
                row['handled'] = True
                reverse_period_indicator_list = ['occurSince', 'occurUntil']
                reverse_period_indicator_list.remove(row['period_indicator'])
                reverse_period_indicator = reverse_period_indicator_list[0]

                end_event = self._find_in_rows(reverse_period_indicator, row, reader.rows(), i)
                if end_event is not None:
                    end_event['handled'] = True
                    row['start_timestamp'] = [r for r in [row, end_event] if r['period_indicator'] == 'occurSince'][0]['timestamp']
                    row['end_timestamp'] = [r for r in [row, end_event] if r['period_indicator'] == 'occurUntil'][0]['timestamp']
                    self._add_element(row['end_timestamp'], self.timestamp_count)
                else:
                    row['start_timestamp'] = row['timestamp']
                    row['end_timestamp'] = "-"
                self._add_element(row['start_timestamp'], self.timestamp_count)
            
            if self.dataset in ['wikidata12k', 'yago11k']:
                self._add_element(row['start_timestamp'], self.timestamp_count)
                self._add_element(row['end_timestamp'], self.timestamp_count)
            
            processed_rows.append(row)

        if self.dataset in ['wikidata11k']:
            for row in processed_rows:
                row.pop('timestamp')
                row.pop('handled')

        self.rows = processed_rows

        for i in ["1", "2", "3"]:
            self._split_once(i, reader)

        if self.dataset in ["wikidata12k", "yago11k"]:
            self._format_original_split(reader)
        
        self.params.timer.stop("split " + self.dataset)

    def _add_element(self, element, element_dir):
        if element not in element_dir.keys():
            element_dir[element] = 1
        else:
            element_dir[element] += 1
    
    def _subtract_element(self, element, element_dir):
        element_dir[element] -= 1

    def _find_in_rows(self, period_indicator, compare_row, rows, start_index):
        for i in list(range(start_index, len(rows))) + list(range(0, start_index+1)):
            row = rows[i]
            if row['head'] == compare_row['head'] and \
                row['relation'] == compare_row['relation'] and \
                row['tail'] == compare_row['tail'] and \
                row['period_indicator'] == period_indicator and \
                row['handled'] == False:
                return row
        return None

    def _include_row(self, row):
        if self.dataset in ['wikidata11k']:
            tail = row['tail']
            return re.match('Q[0-9]+', tail)
        return True

    def _split_once(self, name, reader):
        print("Splitting dataset " + self.dataset + " into train/valid/test set " + name +"...")
        entity_count = copy.copy(self.entitiy_count)
        relation_count = copy.copy(self.relation_count)
        timestamp_count = copy.copy(self.timestamp_count)
        rows = copy.copy(self.rows)
        random.shuffle(rows)

        num_valid = int(len(rows) * self.valid_percentage)
        num_test = int(len(rows) * self.test_percentage)

        for row in rows:
            row['split'] = 'train'

        i = 0
        for row in rows:
            if i % 1000 == 0:
                print(str(num_valid) + " validation and " + str(num_test) + " test facts remaining before completion.")
            i += 1

            if num_valid <= 0 and num_test <= 0:
                break

            if entity_count[row['head']] <= 2 and \
               relation_count[row['relation']] <= 1 and \
               entity_count[row['tail']] <= 2:
                continue
            
            if self.dataset in ['icews14']:
                if timestamp_count[row['timestamp']] <= 1:
                    continue
            
            if self.dataset in ['wikidata11k', 'wikidata12k', 'yago11k']:
                if timestamp_count[row['start_timestamp']] <= 2:
                    continue
                if timestamp_count[row['end_timestamp']] <= 2:
                    continue
            
            self._subtract_element(row['head'], entity_count)
            self._subtract_element(row['relation'], relation_count)
            self._subtract_element(row['tail'], entity_count)

            if self.dataset in ['icews14']:
                self._subtract_element(row['timestamp'], timestamp_count)
            elif self.dataset in ['wikidata11k', 'wikidata12k', 'yago11k']:
                self._subtract_element(row['start_timestamp'], timestamp_count)
                self._subtract_element(row['end_timestamp'], timestamp_count)

            if num_valid > 0:
                row['split'] = 'valid'
                num_valid -= 1
                continue
            if num_test > 0:
                row['split'] = 'test'
                num_test -= 1
                continue
        
        self._write_csv(rows, name, 'train', reader)
        self._write_csv(rows, name, 'valid', reader)
        self._write_csv(rows, name, 'test', reader)
    
    def _format_original_split(self, reader):        
        reader.read_original_splits()

        for row in reader.rows():
            if self.dataset in ['wikidata12k']:
                sliced_date = row['start_timestamp'].split('-')
                year, month, day = sliced_date[:3]
                if year == '####':
                    row['start_timestamp'] = '-'
                else:
                    row['start_timestamp'] = year

                sliced_date = row['end_timestamp'].split('-')
                year, month, day = sliced_date[:3]
                if year == '####':
                    row['end_timestamp'] = '-'
                else:
                    row['end_timestamp'] = year

        self._write_csv(reader.rows(), 'original', 'test', reader)
        self._write_csv(reader.rows(), 'original', 'train', reader)
        self._write_csv(reader.rows(), 'original', 'valid', reader)
            

    def _write_csv(self, rows, split, dataset, reader):
        for formatting in ["A", "B"]:
            text = ""

            if formatting == "B" and self.dataset in ['icews14']:
                continue

            for row in [row for row in rows if row['split'] is dataset]:
                if formatting == "A":
                    if self.dataset in ['icews14']:
                        text = text + row['head'] + "\t" + row['relation'] + "\t" + row['tail'] + "\t" + row['timestamp'] + "\n"
                    if self.dataset in ['wikidata11k', 'wikidata12k', 'yago11k']:
                        text = text + row['head'] + "\t" + row['relation'] + "\t" + row['tail'] + "\t" + row['start_timestamp'] + "\t" + row['end_timestamp'] + "\n"
                    
                if formatting == "B":
                    if self.dataset in ['wikidata11k', 'wikidata12k', 'yago11k']:
                        start_year = row['start_timestamp']
                        if start_year == "-":
                            start_year = "####"

                        end_year = row['end_timestamp']
                        if end_year == "-":
                            end_year = "####"
                        
                        text = text + reader.entity2id(row['head']) + "\t" \
                            + reader.relation2id(row['relation']) + "\t" \
                            + reader.entity2id(row['tail']) + "\t" \
                            + start_year + "-##-##" \
                            + "\t" + end_year + "-##-##" + "\n"
            
            path = os.path.join(self.base_directory, "datasets", self.dataset, "format_" + formatting, "split_" + split, dataset + ".txt")
            write(path, text)


            

