
import argparse

from parameters import Parameters
from rank.ranker import Ranker
from split_dataset.split_dataset import SplitDataset

from statistics.statistics import Statistics
from formatlatex.formatlatex import FormatLatex


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-task', type=str, default='split_dataset', choices=['statistics', 'rank', 'formatlatex', 'split_dataset'])
    parser.add_argument('-dataset', type=str, default='all', choices=['all', 'icews14', 'wikidata11k', 'wikidata12k', 'yago11k'])
    parser.add_argument('-split', type=str, default='all', choices=['all', 'original', '1', '2', '3'])
    parser.add_argument('-embedding', type=str, default='all', choices=['all', 'DE_TransE', 'DE_SimplE', 'DE_DistMult', 'TERO', 'ATISE', 'TFLEX','TimePlex'])
    parser.add_argument('-add_to_result', type=bool, default=True)

    args = parser.parse_args()
    params = Parameters(args)

    if params.embeddings == ['all']:
        params.embeddings = ['DE_TransE']
    if params.datasets == ['all']:
        params.datasets = ['icews14']
    if params.splits == ['all']:
        params.splits = ['original']
    
    match params.task:
        case "split_dataset":
            split_dataset = SplitDataset(params)
            split_dataset.split()
            return 0
        case "rank":
            ranker = Ranker(params)
            ranker.rank()
            return 0
        case "statistics":
            statistics = Statistics(params)
            statistics.run()
            return 0
        case "formatlatex":
            format_latex = FormatLatex(params)
            format_latex.format()
            return 0


if __name__ == '__main__':
    main()


