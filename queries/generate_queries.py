import numpy as np
import pandas as pd
import csv
import re
import json
import os
from scripts import touch

class GenerateQueries():
    def __init__(self, params):
        self.params = params

    # convert txt into csv
    # filename: test.txt
    def _txt_to_csv(self, input_path, output_path):
        with open(output_path, 'w', encoding='utf-8') as f1:
            writer = csv.writer(f1, delimiter='\t')
            with open(input_path, encoding='utf-8') as f2:
                for line in f2:
                    line = re.split("[\t\n]", line)
                    writer.writerow(line)
        print(f"'{input_path}' has been converted to '{output_path}'.")


    # generate corrupted quadruple and save it as csv file
    # filename: test.csv
    # cor_csv_file: corrupted_quadruple_test.csv
    def _generate_corrupted_quadruple(self, input_path, output_path, dataset):
        with open(input_path, encoding='utf-8') as input:
            with open(output_path, 'a', encoding='utf-8') as output:
                output.write("HEAD\tRELATION\tTAIL\tTIME_FROM\tTIME_TO\tANSWER" + '\n')
                for line in input:
                    list = line.strip('\n').split('\t')
                    a = np.array(list)
                    if len(a) <= 1:
                        continue
                    if dataset in ['icews14']:
                        a[4] = '-'
                    for i in range(5):
                        if i == 0:
                            co_qu = np.array([0, a[1], a[2], a[3], a[4], a[0]])
                            output.write('\t'.join(co_qu) + '\n')
                        elif i == 1:
                            co_qu = np.array([a[0], 0, a[2], a[3], a[4], a[1]])
                            output.write('\t'.join(co_qu) + '\n')
                        elif i == 2:
                            co_qu = np.array([a[0], a[1], 0, a[3], a[4], a[2]])
                            output.write('\t'.join(co_qu) + '\n')
                        elif i == 3:
                            co_qu = np.array([a[0], a[1], a[2], 0, a[4], a[3]])
                            output.write('\t'.join(co_qu) + '\n')                            
                        elif dataset in ['wikidata12k', 'yago11k'] and i == 4:
                            co_qu = np.array([a[0], a[1], a[2], a[3], 0, a[4]])
                            output.write('\t'.join(co_qu) + '\n')
        print("Corrupted quadruples are generated, and saved as " + output_path)


    # add ID for each fact
    def _add_fact_id(self, filename, dataset):
        data = pd.read_csv(filename, sep='\t', encoding='utf-8')
        fact_column = []
        num1 = 0
        num2 = 0
        mod = 4 if dataset in ['icews14'] else 5
        for index, row in data.iterrows():
            if index % mod == 0:
                num1 += 1
                num2 = 0
            else:
                pass
            fact_column.append(f'FACT_{num1}_{num2}')
            num2 += 1
        data['FACT_ID'] = fact_column
        data = data[['FACT_ID', 'HEAD', 'RELATION', 'TAIL', 'TIME_FROM', 'TIME_TO', 'ANSWER']]
        data.to_csv(filename, index=False)
        print("FACT_ID has been added.")


    # convert csv into json: [{key: value}, {key: value}...]
    def _csv_to_json(self, input_path, output_path):
        with open(output_path, 'w', encoding='utf-8') as js_f:
            js_f.write('[')
            with open(input_path, encoding='utf-8') as f2:
                records = csv.DictReader(f2)
                first = True
                for row in records:
                    if not first:
                        js_f.write(',')
                    first=False
                    json.dump(row, js_f, indent=4, ensure_ascii=False)
            js_f.write(']')
        print(f"'{input_path}' has been converted to '{output_path}'.")

    def generate_test_quads(self):

        for dataset in self.params.datasets:
            for split in self.params.splits:
                input_path = os.path.join(self.params.base_directory, "datasets", dataset, "format_A", "split_" + split, 'test.txt')
                temp_csv_path = os.path.join(self.params.base_directory, "queries", 'temp.csv')
                temp_test_quads_csv_path = os.path.join(self.params.base_directory, "queries", 'temp_test_quads.csv')
                test_quads_path = os.path.join(self.params.base_directory, "queries", dataset, "split_" + split, 'test_quads.json')

                touch(test_quads_path)

                self._txt_to_csv(input_path, temp_csv_path)
                self._generate_corrupted_quadruple(temp_csv_path, temp_test_quads_csv_path, dataset)
                self._add_fact_id(temp_test_quads_csv_path, dataset)
                self._csv_to_json(temp_test_quads_csv_path, test_quads_path)

                os.remove(temp_csv_path)
                os.remove(temp_test_quads_csv_path)