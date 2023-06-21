
import argparse
import os
import csv
from scripts import touch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-source', type=str, default=R'C:\Users\Jeppe\Documents\Unistuff\Master\P10\datasets\icews14\format_A')

    args = parser.parse_args()
    source = args.source
    dest = os.path.join(os.path.dirname(source), "format_C")
    splits = ["original", "1", "2", "3"]
    datasets = ["test", "train", "valid"]

    for split in splits:
        for dataset in datasets:
            new_facts = []

            input_path = os.path.join(source, "split_" + split, dataset + ".txt")
            with open(input_path, encoding='utf-8') as input:
                facts = csv.DictReader(input, fieldnames=['head', 'relation', 'tail', 'time_from'], delimiter='\t')
                for fact in facts:
                    new_facts.append(fact | {'time_to': fact['time_from']})
            
            output_path = os.path.join(dest, "split_" + split, dataset + ".txt")
            touch(output_path)
            with open(output_path, mode="w", encoding='utf-8', newline='') as output:
                fieldnames = ['head', 'relation', 'tail', 'time_from', 'time_to']
                writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter='\t')
                for fact in new_facts:
                    writer.writerow(fact)

if __name__ == '__main__':
    main()
