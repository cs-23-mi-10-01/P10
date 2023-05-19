
import argparse

from parameters import Parameters
from rank.ranker import Ranker
from split_dataset.split_dataset import SplitDataset
from statistics.statistics import Statistics
from formatlatex.formatlatex import FormatLatex
from queries.generate_queries import GenerateQueries


def main():
    parser = argparse.ArgumentParser()

    #python -task rank -dataset icews14 -embedding DE_TransE -split all
    parser.add_argument('-task', type=str, default='ensemble_naive_voting', choices=['statistics', 'rank', 'formatlatex', 'split_dataset', 'generate_quads', 'best_predictions', 'ensemble_naive_voting', "ensemble_decision_tree"])
    parser.add_argument('-dataset', type=str, default='all', choices=['all', 'icews14', 'wikidata11k', 'wikidata12k', 'yago11k'])
    parser.add_argument('-split', type=str, default='all', choices=['all', 'original', '1', '2', '3'])
    parser.add_argument('-embedding', type=str, default='all', choices=['all','ensemble', 'DE_TransE', 'DE_SimplE', 'DE_DistMult', 'TERO', 'ATISE', 'TFLEX','TimePlex'])

    args = parser.parse_args()
    params = Parameters(args)

    params.timer.start("main")
    if params.embeddings == ['all']:
        params.embeddings = ['DE_TransE', 'DE_SimplE', 'DE_DistMult', 'TERO', 'ATISE','TFLEX', 'TimePlex']
    elif params.embeddings == ['ensemble']:
        params.embeddings = ['DE_TransE', 'DE_SimplE', 'DE_DistMult', 'TERO', 'ATISE']
    if params.datasets == ['all']:
        params.datasets = ['icews14', 'wikidata12k', 'yago11k']
    if params.splits == ['all']:
        params.splits = ['original']
    
    match params.task:
        case "split_dataset":
            split_dataset = SplitDataset(params)
            split_dataset.split()
        case "rank":
            ranker = Ranker(params)
            ranker.rank()
        case "statistics":
            statistics = Statistics(params)
            statistics.run()
        case "formatlatex":
            format_latex = FormatLatex(params)
            format_latex.format()
        case "generate_quads":
            generate_quads = GenerateQueries(params)
            generate_quads.generate_test_quads()
        case "best_predictions":
            ranker = Ranker(params, "best_predictions")
            ranker.rank()
            statistics = Statistics(params)
            statistics.average_timestamp_precision()
        case "ensemble_naive_voting":
            ranker = Ranker(params, "ensemble_naive_voting")
            ranker.rank()
        case "ensemble_decision_tree":
            ranker = Ranker(params, "ensemble_decision_tree")
            ranker.rank()


    params.timer.stop("main")


if __name__ == '__main__':
    main()
