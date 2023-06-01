
import argparse
from scripts import read_json, write_json


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-source_1', type=str, default=R'C:\Users\Jeppe\Documents\Unistuff\Master\P10\data.json')
    parser.add_argument('-target', type=str, default=R'C:\Users\Jeppe\Documents\Unistuff\Master\P10\result\icews14\split_original\ranked_quads.json')

    args = parser.parse_args()
    source_1 = read_json(args.source_1)
    write_json(args.target, source_1)


if __name__ == '__main__':
    main()
