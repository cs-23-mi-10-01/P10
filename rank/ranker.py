
import os
import json

from rank.loader import Loader
from scripts import touch, exists
from rank.de_simple.rank_calculator import RankCalculator as DE_Rank
from rank.TERO.rank_calculator import RankCalculator as TERO_Rank
from rank.TFLEX.rank_calculator import RankCalculator as TFLEX_Rank
from rank.TimePlex.rank_calculator import RankCalculator as TimePlex_Rank

class Ranker:
    def __init__(self, params):
        self.params = params
        self.ranked_quads = []
        self.base_directory = params.base_directory

    def rank(self):
        for dataset in self.params.datasets:
            for split in self.params.splits:
                for embedding_name in self.params.embeddings:
                    results_path = os.path.join(self.base_directory, "result", dataset, "split_" + split, "ranked_quads.json")
                    if exists(results_path):
                        quads_path = results_path
                    else:
                        quads_path = os.path.join(self.base_directory, "queries", dataset, "split_" + split, "test_quads.json")

                    in_file = open(quads_path, "r", encoding="utf8")
                    print("Reading from file " + str(quads_path) + "...")
                    self.ranked_quads = json.load(in_file)
                    in_file.close()
                    
                    model_path = os.path.join(self.base_directory, "models", embedding_name, dataset, "split_" + split, "Model.model")
                    loader = Loader(self.params, dataset, split, model_path, embedding_name)
                    model = loader.load()
                    model.eval()

                    self.ranked_quads = self._generate_ranked_quads(model, embedding_name, dataset, split)

                    touch(results_path)
                    out_file = open(results_path, "w", encoding="utf8")
                    print("Writing to file " + str(results_path) + "...")
                    json.dump(self.ranked_quads, out_file, indent=4)
                    out_file.close()

    def _generate_ranked_quads(self, model, embedding_name, dataset, split):
        ranked_quads = []
        
        if embedding_name in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
            rank_calculator = DE_Rank(self.params, model, dataset)
        if embedding_name in ["TERO", "ATISE"]:
            rank_calculator = TERO_Rank(self.params, model)
        if embedding_name in ["TFLEX"]:
            rank_calculator = TFLEX_Rank(self.params, model)
        if embedding_name in ["TimePlex"]:
            rank_calculator = TimePlex_Rank(self.params, model, dataset)

        for i, quad in zip(range(0, len(self.ranked_quads)), self.ranked_quads):
            if i % 1000 == 0:
                print("Ranking fact " + str(i) + "-" + str(i + 999) + " (total number: " + str(len(self.ranked_quads)) + ") " \
                      + " on dataset " + dataset + ", split " + split + " with embedding " + embedding_name)

            if embedding_name in ["TFLEX"]:
                if not (quad["TAIL"] == "0" or quad["TIME_FROM"] == "0"):
                    ranked_quads.append(quad)
                    continue
            if embedding_name in ['DE_TransE', 'DE_SimplE', 'DE_DistMult']:
                if quad["TIME_TO"] == "0":
                    ranked_quads.append(quad)
                    continue

            ranked_quad = quad
            if "RANK" not in ranked_quad.keys():
                ranked_quad["RANK"] = {}

            ranked_quad["RANK"][embedding_name] = str(rank_calculator.get_rank_of(quad["HEAD"], quad["RELATION"],
                                                                                quad["TAIL"], quad["TIME_FROM"],
                                                                                quad["TIME_TO"], quad["ANSWER"]))
            ranked_quads.append(ranked_quad)

        return ranked_quads

