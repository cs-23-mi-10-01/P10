
import os
import json
from operator import itemgetter
import copy
from rank.loader import Loader
from scripts import touch, exists, read_json
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
                        case "ensemble_naive_voting" | "ensemble_decision_tree":
                            output_path = os.path.join(self.base_directory, "result", dataset, "split_" + split, "ranked_quads.json")
                            
                    
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
                        json_output = self._ensemble_base(dataset, split)
                        

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
                    if self.mode == "ensemble_naive_voting" or "ensemble_decision_tree":
                        break
                    

                            

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

    
    def _ensemble_base(self, dataset, split):
        ranked_quads = []

        
        properties_path = os.path.join(self.base_directory, "statistics", "resources", dataset, "relation_classification_timestamps.json") 
        partitions_path = os.path.join(self.base_directory, "statistics", "resources", dataset, "split_" + split, "partition.json") 
        
        properties = read_json(properties_path)
        partitions = read_json(partitions_path)
        scores = self._ensemble_dict_creator(dataset, split)
        del properties_path
        del partitions_path
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
            # if i < 1500:
            #     continue
            if quad["TIME_TO"] == '0' or quad["ANSWER"] == '-':
                continue
            weight_distribution = {}
            #finding the target
            if quad["HEAD"] == '0':
                target = "head"
            elif quad["RELATION"] == '0':
                target = "relation"
                #removes timeplex cause it can't do relation prediction
                self.params.embeddings.remove("TimePlex")
            elif quad["TAIL"] == '0':
                target = "tail"
                self.params.embeddings.append("TimePlex")
            elif quad["TIME_FROM"] == '0':
                target = "time"

            if self.mode == "ensemble_decision_tree":
                #analyse
                query = self._ensemble_analyser(quad, target, properties, partitions,dataset)
                #decision tree
                weight_distribution = self._ensemble_decision_tree(query, scores, target)
            elif self.mode == "ensemble_naive_voting":
                for embedding_name in self.params.embeddings:
                    weight_distribution[embedding_name] = (1/len(self.params.embeddings))
            #voting
            ranked_quads.append(self._ensemble_voting(dataset,split, quad, rank_calculators, weight_distribution, target))
            print(i," of ",len(self.ranked_quads))
        
        return ranked_quads
    
    def _ensemble_dict_creator(self, dataset, split):
        scores = {}

        head_path = os.path.join(self.base_directory, "result", dataset, "split_" + split,"semester_9_hypothesis_1", "head.json")
        relation_path = os.path.join(self.base_directory, "result", dataset, "split_" + split,"semester_9_hypothesis_1", "relation.json") 
        tail_path = os.path.join(self.base_directory, "result", dataset, "split_" + split,"semester_9_hypothesis_1", "tail.json") 
        time_path = os.path.join(self.base_directory, "result", dataset, "split_" + split,"semester_9_hypothesis_1", "time_from.json")  
        overall_path = os.path.join(self.base_directory, "result", dataset, "split_" + split, "overall_scores.json")
        time_density_path = os.path.join(self.base_directory, "result", dataset, "split_" + split, "time_density.json") 
        anti_symmetry_path = os.path.join(self.base_directory, "result", dataset, "split_" + split,"semester_10_hypothesis_3", "relation_property_anti-symmetric_timestamps.json")  
        inverse_path = os.path.join(self.base_directory, "result", dataset, "split_" + split,"semester_10_hypothesis_3", "relation_property_inverse_timestamps.json")  
        reflexive_path = os.path.join(self.base_directory, "result", dataset, "split_" + split,"semester_10_hypothesis_3", "relation_property_reflexive_timestamps.json")  
        symmetry_path = os.path.join(self.base_directory, "result", dataset, "split_" + split,"semester_10_hypothesis_3", "relation_property_symmetric_timestamps.json")  

        head = read_json(head_path)
        relaition = read_json(relation_path)
        tail = read_json(tail_path)
        time = read_json(time_path)
        overall = read_json(overall_path)
        
        

        time_density = read_json(time_density_path)

        symmetry = read_json(symmetry_path)
        inverse = read_json(inverse_path)
        anti_symmetry = read_json(anti_symmetry_path)
        reflexive = read_json(reflexive_path)

        #removing TFLEX from all datasets
        overall.pop('TFLEX', None)
        head.pop('TFLEX', None)
        relaition.pop('TFLEX', None)
        tail.pop('TFLEX', None)
        time.pop('TFLEX', None)
        time_density["dense"].pop('TFLEX', None)
        time_density["sparse"].pop('TFLEX', None)
        symmetry["symmetric"].pop('TFLEX', None)
        symmetry["not symmetric"].pop('TFLEX', None)
        inverse["inverse"].pop('TFLEX', None)
        inverse["not inverse"].pop('TFLEX', None)
        anti_symmetry["anti-symmetric"].pop('TFLEX', None)
        anti_symmetry["not anti-symmetric"].pop('TFLEX', None)
        reflexive["not reflexive"].pop('TFLEX', None)
        reflexive["reflexive"].pop('TFLEX', None)

        scores["head"] = head
        scores["relation"] = relaition
        scores["tail"] = tail
        scores["time"] = time
        scores["overall"] = overall
        scores = scores | time_density | symmetry | inverse | anti_symmetry | reflexive
        scores.pop("facts_in_class")
        scores.pop("facts_not_in_class")
    
        return scores
    
    def _ensemble_analyser(self, quad, target, properties, partitions,dataset):
        query = {"properties": [], "density": ''}
        if target != "relation":   
            for lol in properties[quad["RELATION"]]:
                 if properties[quad["RELATION"]][lol] == True:
                     query["properties"] += [lol]

        if target != "time" and quad["TIME_FROM"] != "-":

            if dataset in ["wikidata12k", "yago11k"]:
                quad_start_date = f"""{int(quad["TIME_FROM"]):04d}-01-01"""
            else:
                quad_start_date = quad["TIME_FROM"]

            for partition in partitions:
                if partition["start_date"] <= quad_start_date and \
                    quad_start_date < partition["end_date"]:
                    
                    if partition["partition"] == "sparse":
                        query["density"] = "sparse"
                    if partition["partition"] == "dense":
                        query["density"] = "dense"

        return query

    def _ensemble_decision_tree(self, query, scores, target):
        weight_distribution = {}
        if target == "relation":
            relation_scores = copy.deepcopy(scores)
            for r in scores:
                relation_scores[r].pop("TimePlex", None)
                scores = relation_scores
                

        diff =self.diff_min_max_finder(scores["overall"])
        normalized = self.mrr_normalizer(scores["overall"])
        for method in normalized:
            weight_distribution[method] = normalized[method] * diff

        for p in query["properties"]:
            diff =self.diff_min_max_finder(scores[p])
            normalized = self.mrr_normalizer(scores[p])
            for method in normalized:
                weight_distribution[method] += normalized[method] * diff

        if query["density"] == "dense":
            diff =self.diff_min_max_finder(scores["dense"])
            normalized = self.mrr_normalizer(scores["dense"])
            for method in normalized:
                weight_distribution[method] += normalized[method] * diff

        elif query["density"] == "sparse":
            diff =self.diff_min_max_finder(scores["sparse"])
            normalized = self.mrr_normalizer(scores["sparse"])
            for method in normalized:
                weight_distribution[method] += normalized[method] * diff
            
        match(target):
            case("head"):
                diff =self.diff_min_max_finder(scores["head"])
                normalized = self.mrr_normalizer(scores["head"])
                for method in normalized:
                    weight_distribution[method] += normalized[method] * diff

            case("relation"):
                diff =self.diff_min_max_finder(scores["relation"])
                normalized = self.mrr_normalizer(scores["relation"])
                for method in normalized:
                    weight_distribution[method] += normalized[method] * diff

            case("tail"):
                diff =self.diff_min_max_finder(scores["tail"])
                normalized = self.mrr_normalizer(scores["tail"])
                for method in normalized:
                    weight_distribution[method] += normalized[method] * diff
                    
            case("time"):
                diff =self.diff_min_max_finder(scores["time"])
                normalized = self.mrr_normalizer(scores["time"])
                for method in normalized:
                    weight_distribution[method] += normalized[method] * diff

        #normalized the final weights so the sum 1
        factor=1.0/sum(weight_distribution.values())
        for k in weight_distribution:
            weight_distribution[k] = weight_distribution[k]*factor
        return weight_distribution
    
    def _ensemble_voting(self,dataset,split, quad, rank_calculators,weight_distribution, target):
        voting_points = {}
        answer_Id = int
        for embedding_name in self.params.embeddings:
            if embedding_name == "TimePlex":
                timeplex_scores = rank_calculators[embedding_name].simulate_fact_scores(quad["HEAD"], quad["RELATION"],
                                                quad["TAIL"], quad["TIME_FROM"],
                                                quad["TIME_TO"], quad["ANSWER"])
                fact_scores = []
                for key in timeplex_scores.keys():
                    fact_scores.append([key[0],key[1],key[2],key[3],timeplex_scores[key]])

            else:
                fact_scores = rank_calculators[embedding_name].simulate_fact_scores(quad["HEAD"], quad["RELATION"],
                                                quad["TAIL"], quad["TIME_FROM"],
                                                quad["TIME_TO"], quad["ANSWER"])
            #the first element of fact scores is the correct answer
            if embedding_name != "TimePlex":
                del(fact_scores[0])
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
                        if dataset in ['wikidata12k', 'yago11k']:
                            if quad["ANSWER"] == "-":
                                answer = quad["ANSWER"]
                            elif int(quad["ANSWER"]) < 1000:
                                answer = "0"+ quad["ANSWER"] + "-01-01"
                            else:
                                answer = quad["ANSWER"] + "-01-01"
                        else:
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

        ranked_quad = quad
        if "RANK" not in ranked_quad.keys():
            ranked_quad["RANK"] = {}

        ranked_quad["RANK"][str(self.mode)] = str(rank)
        return ranked_quad
        
    def diff_min_max_finder(self, dict):
        #finds the difference between max and min in given dict
        min = dict["DE_TransE"]["MRR"]
        max = dict["DE_TransE"]["MRR"]
        for methods in dict:
            if dict[methods]["MRR"] > max:
                max = dict[methods]["MRR"]
            if dict[methods]["MRR"] < min:
                min = dict[methods]["MRR"]
        
        return max - min
    
    def mrr_normalizer(self, dict):
        #normalized the given dict so the max value = 1 and min value = 0
        normalized_dict = {}
        norm_list = []
        for method in dict:
            norm_list.append(dict[method]["MRR"])
        norm_list = [(float(i)-min(norm_list))/(max(norm_list)-min(norm_list)) for i in norm_list]
        for i, method in zip(range(0, len(norm_list)), dict):
            normalized_dict[method] = norm_list[i]
           

        return normalized_dict