
import os
import json

from rank.loader import Loader
from scripts import touch
from rank.de_simple.rank_calculator import RankCalculator as DE_Rank
from rank.TERO.rank_calculator import RankCalculator as TERO_Rank
from rank.TFLEX.rank_calculator import RankCalculator as TFLEX_Rank
from rank.TimePlex.rank_calculator import RankCalculator as TimePlex_Rank

class Ranker:
    def __init__(self, params):
        self.params = params
        self.ranked_quads = []
        self.base_directory = params.base_directory
        self.dataset = params.dataset

        if not self.params.add_to_result:
            self.quads_path = os.path.join(self.base_directory, "queries", self.dataset, "corrupted_quads.json")
        else:
            self.quads_path = os.path.join(self.base_directory, "result", self.dataset, "ranked_quads.json")
        
        self.embeddings = self.params.embeddings

    def rank(self):
        in_file = open(self.quads_path, "r", encoding="utf8")
        self.ranked_quads = json.load(in_file)
        in_file.close()         

        for embedding_name in self.embeddings:
            model_path = os.path.join(self.base_directory, "models", embedding_name, self.dataset, "Model.model")
            loader = Loader(self.params, model_path, embedding_name)
            model = loader.load()
            model.eval()

            self.ranked_quads = self._generate_ranked_quads(model, embedding_name)
        
        results_path = os.path.join(self.base_directory, "result", self.dataset, "ranked_quads.json")

        touch(results_path)
        out_file = open(results_path, "w", encoding="utf8")
        json.dump(self.ranked_quads, out_file, indent=4)
        out_file.close()

    def _generate_ranked_quads(self, model, embedding_name):
        ranked_quads = []
        
        if embedding_name in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
            rank_calculator = DE_Rank(self.params, model)
        if embedding_name in ["TERO", "ATISE"]:
            rank_calculator = TERO_Rank(self.params, model)
        if embedding_name in ["TFLEX"]:
            rank_calculator = TFLEX_Rank(self.params, model)
        if embedding_name in ["TimePlex"]:
            rank_calculator = TimePlex_Rank(self.params, model)

        for i, quad in zip(range(0, len(self.ranked_quads)), self.ranked_quads):
            if i % 100 == 0:
                print("Ranking fact " + str(i) + "-" + str(i + 99) + " (total number: " + str(len(self.ranked_quads)) + ") with embedding " + embedding_name)

            if embedding_name in ["TFLEX"]:
                if not (quad["TAIL"] == "0" or quad["TIME"] == "0"):
                    ranked_quads.append(quad)
                    continue

            ranked_quad = quad
            if "RANK" not in ranked_quad.keys():
                ranked_quad["RANK"] = {}

            ranked_quad["RANK"][embedding_name] = str(rank_calculator.get_rank_of(quad["HEAD"], quad["RELATION"],
                                                                                       quad["TAIL"], quad["TIME"],
                                                                                       quad["ANSWER"]))
            ranked_quads.append(ranked_quad)

        return ranked_quads

