
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
    parser.add_argument('-task', type=str, default='statistics', choices=['statistics', 'rank', 'formatlatex', 'split_dataset', 'generate_quads', 'best_predictions', 'ensemble_naive_voting', "ensemble_decision_tree", "ablation_overall", "ablation_property", "ablation_false_property", "ablation_time_density", "ablation_target", "ablation_no_property", "ablation_one_forth_property", "ablation_only_property", "ablation_only_target", "ablation_only_overall", "ablation_only_time_density"])
    parser.add_argument('-dataset', type=str, default='all', choices=['all', 'icews14', 'wikidata11k', 'wikidata12k', 'yago11k'])
    parser.add_argument('-split', type=str, default='original', choices=['all', 'original', '1', '2', '3'])
    parser.add_argument('-embedding', type=str, default='overall_scores', choices=['all','ensemble', 'DE_TransE', 'DE_SimplE', 'DE_DistMult', 'TERO', 'ATISE', 'TFLEX','TimePlex', 'overall_scores'])

    args = parser.parse_args()
    params = Parameters(args)

    params.timer.start("main")
    if params.embeddings == ['all']:
        params.embeddings = ['DE_TransE', 'DE_SimplE', 'DE_DistMult', 'TERO', 'ATISE', 'TimePlex']
    elif params.embeddings == ['ensemble']:
        params.embeddings = ['DE_TransE', 'DE_SimplE', 'DE_DistMult', 'TERO', 'ATISE', 'TimePlex']
    elif params.embeddings == ['overall_scores']:
        params.embeddings = ['DE_TransE', 'DE_SimplE', 'DE_DistMult', 'TERO', 'ATISE',  'TimePlex','ensemble_naive_voting', 'ensemble_decision_tree', "ablation_overall", "ablation_property", "ablation_false_property", "ablation_time_density", "ablation_target" ,"ablation_no_property", "ablation_one_forth_property", "ablation_only_property", "ablation_only_target", "ablation_only_overall", "ablation_only_time_density"]
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
            format_latex = FormatLatex(params, task=["overall_scores"])
            format_latex.format()
        case "generate_quads":
            generate_quads = GenerateQueries(params)
            generate_quads.generate_test_quads()
        case "best_predictions":
            ranker = Ranker(params, "best_predictions", True)
            ranker.rank()
            statistics = Statistics(params)
            statistics.average_timestamp_precision()
            format_latex = FormatLatex(params, ["time_prediction_mae"])
            format_latex.format()
        case "ensemble_naive_voting":
            ranker = Ranker(params, "ensemble_naive_voting")
            ranker.rank()
        case task if task in ["ensemble_decision_tree", "ablation_overall", "ablation_property", "ablation_false_property", "ablation_time_density", "ablation_target", "ablation_no_property", "ablation_one_forth_property", "ablation_only_property", "ablation_only_target", "ablation_only_overall", "ablation_only_time_density"]:
            ranker = Ranker(params, "ensemble_decision_tree")
            ranker.rank()

            

    params.timer.stop("main")


if __name__ == '__main__':
    main()
