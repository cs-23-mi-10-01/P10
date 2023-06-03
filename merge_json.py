
import argparse
from scripts import read_json, write_json


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-source_1', type=str, default=R'C:\Users\Jeppe\Documents\Unistuff\Master\P10\result\icews14\split_original\ranked_quads.json')
    parser.add_argument('-source_2', type=str, default=R'C:\Users\Jeppe\Documents\Unistuff\Master\P10\result\icews14\split_original\ranked_quads_ablation_no_reflexion.json')
    parser.add_argument('-output_path', type=str, default=R'C:\Users\Jeppe\Documents\Unistuff\Master\P10\result\icews14\split_original\ranked_quads.json')
    parser.add_argument('-override_method', type=str, default='')

    args = parser.parse_args()
    source_1 = read_json(args.source_1)
    source_2 = read_json(args.source_2)

    for quad_1, quad_2 in zip(source_1, source_2):
        if "RANK" not in quad_1.keys():
            continue
        if "RANK" not in quad_2.keys():
            continue

        if args.override_method == '':
            quad_1["RANK"] = quad_1["RANK"] | quad_2["RANK"]
        else:
            quad_1["RANK"][args.override_method] = quad_2["RANK"][args.override_method]

    write_json(args.output_path, source_1)


if __name__ == '__main__':
    main()
