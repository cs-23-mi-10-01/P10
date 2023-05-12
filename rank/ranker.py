
import os
import json
from operator import itemgetter

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
                        case "ensemble_naive_voting":
                            output_path = os.path.join(self.base_directory, "result", dataset, "split_" + split, "ensemble_naive_voting.json")
                            
                    
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

                    if self.mode == "ensemble_naive_voting" or "ensemble_decision_tree":
                        self._ensemble_base(dataset, split)
                        break

                    else:
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
                            rank_calculator = TimePlex_Rank(self.params, model)
                        
                        # write to file
                        json_output = generate_function(rank_calculator, embedding_name, dataset, split)
                    touch(output_path)
                    out_file = open(output_path, "w", encoding="utf8")
                    print("Writing to file " + str(output_path) + "...")
                    json.dump(json_output, out_file, indent=4)
                    out_file.close()

                    #Ensemble goes through all the selected methods but only need to do it once. 
                    

                            

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
            ranked_quad["RANK"][embedding_name] = str(rank_calculator.rank_of_correct_prediction(fact_scores))
            ranked_quads.append(ranked_quad)

        return ranked_quads
    
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
    
    def _ensemble_base(self, dataset, split):
        ensemble_scores = []
        #Goes through each model one fact at a time and the ensemble method combines all the answers into it's final answer, which is then save to ensemble_scores
        # load model via torch
        rank_calculators = {}
        for embedding_name in self.params.embeddings:
            print(embedding_name, "loaded")
            model_path = os.path.join(self.base_directory, "models", embedding_name, dataset, "split_" + split, "Model.model")
            loader = Loader(self.params, dataset, split, model_path, embedding_name)
            model = loader.load()
            model.eval()
            
            
            # load all the methods into rank calculators
            if embedding_name in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
                rank_calculators[embedding_name] = DE_Rank(self.params, model, dataset)
            if embedding_name in ["TERO", "ATISE"]:
                rank_calculators[embedding_name] = TERO_Rank(self.params, model, dataset)
            if embedding_name in ["TimePlex"]:
                rank_calculators[embedding_name] = TimePlex_Rank(self.params, model, dataset)
        
        for i, quad in zip(range(0, len(self.ranked_quads)), self.ranked_quads):
            weight_distribution = {}
            #finding the target
            if i % 4 == 0:
                target = "head"
            elif i % 4 == 1:
                target = "relation"
            if i % 4 == 2:
                target = "tail"
            if i % 4 == 3:
                target = "time"
            #analyse
            if self.mode == "ensemble_decision_tree":
                pass
            #decision tree
            elif self.mode == "ensemble_naive_voting":
                for embedding_name in self.params.embeddings:
                    weight_distribution[embedding_name] = (1/len(self.params.embeddings))
            #voting
            self._ensemble_voting(dataset,split, quad, rank_calculators, weight_distribution, target)
            
        return ensemble_scores
    
    
    def _ensemble_voting(self,dataset,split, quad, rank_calculators,weight_distribution, target):
        voting_points = {}
        answer_Id = int
        for embedding_name in self.params.embeddings:
            
            fact_scores = rank_calculators[embedding_name].simulate_fact_scores(quad["HEAD"], quad["RELATION"],
                                                quad["TAIL"], quad["TIME_FROM"],
                                                quad["TIME_TO"], quad["ANSWER"])
            #the first element of fact scores is the correct answer
            fact_scores[0]
            for i in range(0, len(fact_scores)):
                #combines the date into single element in the DE models for consistency with tero and atise
                if embedding_name in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
                    fact_scores[i][3] = f"{fact_scores[i][3]:04d}-{fact_scores[i][4]:02d}-{fact_scores[i][5]:02d}"
                    del fact_scores[i][4:6] 
                #creates id in each list in element 5
                fact_scores[i].append(i)

            sorted_fact_scores = sorted(fact_scores, key=itemgetter(4), reverse=True)

            for i in range(0, len(sorted_fact_scores)):
                
                if sorted_fact_scores[i][5] not in voting_points:
                    voting_points[sorted_fact_scores[i][5]] = 0
                
                voting_points[sorted_fact_scores[i][5]] += (1/(i+1)) * weight_distribution[embedding_name]
            
            if(embedding_name == self.params.embeddings[0]):
                #print(embedding_name)
                match(target):
                    case "head":
                        check = 0
                        answer =rank_calculators[self.params.embeddings[0]].get_ent_id(quad["ANSWER"])
                    case "relation":
                        check = 1
                        answer =rank_calculators[self.params.embeddings[0]].get_rel_id(quad["ANSWER"])
                    case "tail":
                        check = 2
                        answer =rank_calculators[self.params.embeddings[0]].get_ent_id(quad["ANSWER"])
                    case "time":
                        check = 3
                        #don't need conversion here as time is already in the right format
                        answer = quad["ANSWER"]
                for simu_fact in sorted_fact_scores:

                        if simu_fact[check] == answer:

                            answer_Id = simu_fact[5]
                            break
                            

                            
        
        #finding rank of correct answer
        rank = 1
        for (k,v) in voting_points.items():
            if v > voting_points[answer_Id]:
                rank += 1

        print(target)
        print(rank)
        
