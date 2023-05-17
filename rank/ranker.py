
import os
import json

from rank.loader import Loader
from scripts import touch, exists
from rank.de_simple.rank_calculator import RankCalculator as DE_Rank
from rank.TERO.rank_calculator import RankCalculator as TERO_Rank
from rank.TFLEX.rank_calculator import RankCalculator as TFLEX_Rank
from rank.TimePlex.rank_calculator import RankCalculator as TimePlex_Rank

class Ranker:
    def __init__(self, params, mode = 'rank'):
        self.params = params
        self.ranked_quads = []
        self.base_directory = params.base_directory
        self.mode = mode

    def rank(self):
        for dataset in self.params.datasets:
            for split in self.params.splits:
                for embedding_name in self.params.embeddings:
                    
                    # set output paths and functions for different modes
                    match(self.mode):
                        case "rank":
                            output_path = os.path.join(self.base_directory, "result", dataset, "split_" + split, "ranked_quads.json")
                            generate_function = getattr(self, '_generate_ranked_quads')
                        case "best_predictions":
                            output_path = os.path.join(self.base_directory, "result", dataset, "split_" + split, "best_predictions.json")
                            generate_function = getattr(self, '_generate_best_predictions')
                    
                    #set input path
                    if exists(output_path):
                        quads_path = output_path
                    else:
                        quads_path = os.path.join(self.base_directory, "queries", dataset, "split_" + split, "test_quads.json")                  
                    
                    # read from input
                    in_file = open(quads_path, "r", encoding="utf8")
                    print("Reading from file " + str(quads_path) + "...")
                    self.ranked_quads = json.load(in_file)
                    in_file.close()

                    # load model via torch
                    model_path = os.path.join(self.base_directory, "models", embedding_name, dataset, "split_" + split, "Model.model")
                    loader = Loader(self.params, dataset, split, model_path, embedding_name)
                    model = loader.load()
                    model.eval()
                    
                    # select rank calculator depending on method
                    if embedding_name in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
                        rank_calculator = DE_Rank(self.params, model, dataset)
                    if embedding_name in ["TERO", "ATISE"]:
                        rank_calculator = TERO_Rank(self.params, model, dataset)
                    if embedding_name in ["TFLEX"]:
                        rank_calculator = TFLEX_Rank(self.params, model)
                    if embedding_name in ["TimePlex"]:
                        rank_calculator = TimePlex_Rank(self.params, model, dataset)
                    
                    # write to file
                    json_output = generate_function(rank_calculator, embedding_name, dataset, split)
                    touch(output_path)
                    out_file = open(output_path, "w", encoding="utf8")
                    print("Writing to file " + str(output_path) + "...")
                    json.dump(json_output, out_file, indent=4)
                    out_file.close()

                            

    def _generate_ranked_quads(self, rank_calculator, embedding_name, dataset, split):
        ranked_quads = []

        for i, quad in zip(range(0, len(self.ranked_quads)), self.ranked_quads):
            if i % 1000 == 0:
                print("Ranking fact " + str(i) + "-" + str(i + 999) + " (total number: " + str(len(self.ranked_quads)) + ") " \
                      + " on dataset " + dataset + ", split " + split + " with embedding " + embedding_name)

            if embedding_name in ["TFLEX"]:
                if not (quad["TAIL"] == "0" or quad["TIME_FROM"] == "0"):
                    ranked_quads.append(quad)
                    continue
            if embedding_name in ['DE_TransE', 'DE_SimplE', 'DE_DistMult', 'TERO', 'ATISE']:
                if quad["TIME_TO"] == "0":
                    ranked_quads.append(quad)
                    continue
            if embedding_name in ["TimePlex"]:
                if quad["RELATION"] == "0":
                    ranked_quads.append(quad)
                    continue

            ranked_quad = quad
            if "RANK" not in ranked_quad.keys():
                ranked_quad["RANK"] = {}

            fact_scores = rank_calculator.simulate_fact_scores(quad["HEAD"], quad["RELATION"],
                                                    quad["TAIL"], quad["TIME_FROM"],
                                                    quad["TIME_TO"], quad["ANSWER"])
            ranked_quad["RANK"][embedding_name] = str(rank_calculator.rank_of_correct_prediction(fact_scores, self._correct_fact(quad)))
            ranked_quads.append(ranked_quad)

        return ranked_quads
    
    def _correct_fact(self, quad):
        if quad["HEAD"] == "0":
            return (quad["ANSWER"], quad["RELATION"], quad["TAIL"], quad["TIME_FROM"], quad["TIME_TO"])
        if quad["RELATION"] == "0":
            return (quad["HEAD"], quad["ANSWER"], quad["TAIL"], quad["TIME_FROM"], quad["TIME_TO"])
        if quad["TAIL"] == "0":
            return (quad["HEAD"], quad["RELATION"], quad["ANSWER"], quad["TIME_FROM"], quad["TIME_TO"])
        if quad["TIME_FROM"] == "0":
            return (quad["HEAD"], quad["RELATION"], quad["TAIL"], quad["ANSWER"], quad["TIME_TO"])
        if quad["TIME_TO"] == "0":
            return (quad["HEAD"], quad["RELATION"], quad["TAIL"], quad["TIME_FROM"], quad["ANSWER"])

    def _generate_best_predictions(self, rank_calculator, embedding_name, dataset, split):
        best_predictions = []
        for i, quad in zip(range(0, len(self.ranked_quads)), self.ranked_quads):
            if i % 1000 == 0:
                print("Generating predictions for fact " + str(i) + "-" + str(i + 999) + " (total number: " + str(len(self.ranked_quads)) + ") " \
                      + " on dataset " + dataset + ", split " + split + " with embedding " + embedding_name)

            if quad["TIME_FROM"] == "0":
                best_prediction_quad = quad
                if "RANK" in best_prediction_quad.keys():
                    best_prediction_quad.pop("RANK")
                if "BEST_PREDICTION" not in best_prediction_quad.keys():
                    best_prediction_quad["BEST_PREDICTION"] = {}
                if embedding_name not in best_prediction_quad["BEST_PREDICTION"].keys():
                    best_prediction_quad["BEST_PREDICTION"][embedding_name] = {}
                else:
                    best_predictions.append(quad)
                    continue
                
                fact_scores = rank_calculator.simulate_fact_scores(quad["HEAD"], quad["RELATION"],
                                                    quad["TAIL"], quad["TIME_FROM"],
                                                    quad["TIME_TO"], quad["ANSWER"])
                best_prediction_quad["BEST_PREDICTION"][embedding_name]["PREDICTION"] = rank_calculator.best_prediction(fact_scores)
                best_predictions.append(best_prediction_quad)

        return best_predictions
