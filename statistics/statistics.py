
import os
import json
import pandas

from datetime import date
from scripts import touch, year_to_iso_format, read_json, write_json, exists, write, getdctval
from statistics.measure import Measure
from copy import deepcopy
from dataset_handler.dataset_handler import DatasetHandler
from statistics.semester_10_time_density_hypothesis import TimeDensityHypothesis
from statistics.semester_10_relation_properties_hypothesis import RelationPropertiesHypothesis
from statistics.semester_10_voting_hypothesis import VotingHypothesis
from rank.ranker import Ranker
from dataset_handler.dataset_handler import DatasetHandler


class Statistics():
    def __init__(self, params) -> None:
        self.params = params
        self.gamma = 15
        self.zeta = 5
    
    def read_csv(self, path):
        print("Reading from file " + path + "...")
        in_file = open(path, "r", encoding="utf8")
        csv = pandas.read_csv(in_file, delimiter='\t')
        in_file.close()
        return csv
    
    def calculate_overall_scores(self, ranked_quads, embeddings, dataset, split):
        print("Rank of all question tuples:")
        
        measure = Measure()

        for quad in ranked_quads:
            # if not (quad["TAIL"] == "0" or quad["HEAD"] == "0"):
            #     continue

            ranks = {}
            for embedding in embeddings:
                if "RANK" not in quad.keys():
                    continue
                if embedding not in quad["RANK"].keys():
                    continue
                if embedding == "TFLEX":
                    if not (quad["TAIL"] == "0" or quad["TIME_FROM"] == "0"):
                        continue
                if embedding in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
                    if quad["TIME_TO"] == "0":
                        continue
                
                ranks[embedding] = int(float(quad["RANK"][embedding]))
            measure.update(ranks)
        
        measure.normalize()
        measure.print()

        results_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "overall_scores.json")
        write_json(results_path, measure.as_dict())

    def semester_9_hypothesis_1(self, ranked_quads, embeddings, dataset, split, normalization_scores = None):
        for element_type in ["HEAD", "RELATION", "TAIL", "TIME_FROM"]:
            print("Rank of question tuples when " + str(element_type) + " is the answer element:")
            
            measure = Measure()

            for quad in ranked_quads:
                if quad[element_type] != "0":
                    continue

                ranks = {}
                for embedding in embeddings:
                    if embedding == "TFLEX":
                        if element_type not in ["TAIL", "TIME_FROM"]:
                            continue
                    if embedding in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
                        if quad["TIME_TO"] == "0":
                            continue
                    
                    if embedding in quad["RANK"].keys():
                        ranks[embedding] = int(float(quad["RANK"][embedding]))
                measure.update(ranks)
            
            measure.print()
            measure.normalize()

            results_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_9_hypothesis_1", str(element_type).lower()+".json")
            write_json(results_path, measure.as_dict())

            if normalization_scores is not None:
                measure.normalize_to(normalization_scores)
                results_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_9_hypothesis_1", str(element_type).lower()+"_normalized.json")
                write_json(results_path, measure.as_dict())

    def semester_9_hypothesis_2(self, ranked_quads, embeddings, dataset, split, normalization_scores = None):
        for element in ["ENTITY", "RELATION", "TIME"]:
            print("Testing hypothesis 2 on " + str(element) + "s:")
            element_measures = {}
            json_output = []
            json_output_normalized = []

            if element == "ENTITY":
                target_parts = ["HEAD", "TAIL"]
            elif element == "TIME":
                target_parts = ["TIME_FROM", "TIME_TO"]
            else:
                target_parts = [element]

            for target_part in target_parts:
                for quad in ranked_quads:
                    if quad[target_part] == "0":
                        continue

                    if quad[target_part] not in element_measures.keys():
                        element_measures[quad[target_part]] = Measure()
                    
                    ranks = {}
                    for embedding in embeddings:
                        if embedding == "TFLEX":
                            if not (quad["TAIL"] == "0" or quad["TIME_FROM"] == "0"):
                                continue
                        if embedding in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
                            if quad["TIME_TO"] == "0":
                                continue

                        ranks[embedding] = int(float(quad["RANK"][embedding]))
                    element_measures[quad[target_part]].update(ranks)
            
            for element_key in element_measures.keys():
                element_measures[element_key].normalize()

                json_output.append({"ELEMENT": element_key, "NUM_FACTS": max(element_measures[element_key].num_facts.values()), "MEASURE": element_measures[element_key].as_dict()})
                if normalization_scores is not None:
                    element_measures[element_key].normalize_to(normalization_scores)
                    json_output_normalized.append({"ELEMENT": element_key, "NUM_FACTS": max(element_measures[element_key].num_facts.values()), "MEASURE": element_measures[element_key].as_dict()})

            pop_indexes = []
            for i, dict in enumerate(json_output):
                if dict["NUM_FACTS"] < self.gamma:
                    pop_indexes.append(i)
                
            for i in range(len(pop_indexes) -1, -1, -1):
                json_output.pop(pop_indexes[i])
                if normalization_scores is not None:
                    json_output_normalized.pop(pop_indexes[i])

            json_output.sort(key=lambda val: val["NUM_FACTS"], reverse=True)
            if normalization_scores is not None:
                json_output_normalized.sort(key=lambda val: val["NUM_FACTS"], reverse=True)

            results_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_9_hypothesis_2", str(element).lower()+".json")
            write_json(results_path, json_output)
            
            if normalization_scores is not None:
                results_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_9_hypothesis_2", str(element).lower()+"_normalized.json")
                write_json(results_path, json_output_normalized)

    def calc_overlap(self, arr_x, arr_y):
        shared_vals = 0.0
        total_vals = 0.0

        for val in arr_x:
            if val in arr_y:
                shared_vals += 1.0
            total_vals += 1.0
        
        if total_vals == 0:
            return 0
        return shared_vals/total_vals

    def semester_9_hypothesis_2_top_x(self, embeddings, dataset, split, top_num = 20, percentage = False):
        for element in ["ENTITY", "RELATION", "TIME"]:
            print("Testing hypothesis 2 top X percent on " + str(element) + "s:")

            input_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_9_hypothesis_2", str(element).lower()+".json")
            json_input = read_json(input_path)

            no_of_elements = len(json_input)

            if percentage:
                top_percentage = float(top_num) / 100.0
                element_split = int(no_of_elements * top_percentage)
            else:
                element_split = top_num

            json_top = {}

            for embedding in embeddings:
                if embedding not in json_top.keys():
                    json_top[embedding] = {"TOP": []}

                json_input.sort(key=lambda val: val["MEASURE"][embedding]["MRR"], reverse=True)
                for i in range(0, no_of_elements):
                    if i < element_split:
                        json_top[embedding]["TOP"].append(json_input[i]["ELEMENT"])
            
            json_overlap = []

            for embedding_n in embeddings:
                for embedding_m in embeddings:
                    json_overlap.append({
                        "EMBEDDING_N": embedding_n,
                        "EMBEDDING_M": embedding_m,
                        "OVERLAP_TOP": self.calc_overlap(json_top[embedding_n]["TOP"], json_top[embedding_m]["TOP"])
                    })

            results_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_9_hypothesis_2", 
                                        "top_x_overlap", str(element).lower()+"_top_" + str(top_num) + ("pct" if percentage else "") + ".json")
            write_json(results_path, json_top)
            
            overlap_results_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_9_hypothesis_2", 
                                        "top_x_overlap", str(element).lower()+"_top_" + str(top_num) + ("pct" if percentage else "") + "_overlap.json")
            write_json(overlap_results_path, json_overlap)
                
    def semester_9_hypothesis_3(self, ranked_quads, embeddings, dataset, split, normalization_scores = None):
        entity_measures = {}
        print("Testing hypothesis 3.")

        for quad in ranked_quads:
            if quad["HEAD"] == "0" or quad["TAIL"] == "0":
                continue

            entity_n = quad["HEAD"]
            entity_m = quad["TAIL"]
            key = entity_n+";"+entity_m

            if key not in entity_measures.keys():
                entity_measures[key] = {"ENTITY_N": entity_n, "ENTITY_M": entity_m, "FACTS": 0, "RANK": Measure()}
            
            ranks = {}
            for embedding in embeddings:
                if embedding == "TFLEX":
                    if not (quad["TAIL"] == "0" or quad["TIME_FROM"] == "0"):
                        continue
                if embedding in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
                    if quad["TIME_TO"] == "0":
                        continue

                ranks[embedding] = int(float(quad["RANK"][embedding]))
            entity_measures[key]["RANK"].update(ranks)
            entity_measures[key]["FACTS"] += 1
        
        for key in entity_measures.keys():
            entity_measures[key]["RANK"].normalize()
        
        for key in entity_measures.keys():
            entity_n = entity_measures[key]["ENTITY_N"]
            entity_m = entity_measures[key]["ENTITY_M"]
            other_key = entity_m+";"+entity_n
            if other_key in entity_measures.keys():
                if entity_measures[key]["FACTS"] >= self.zeta and entity_measures[other_key]["FACTS"] >= self.zeta:
                    entity_measures[key]["diff_list"] = {}
                    for embedding in embeddings:
                        entity_measures[key]["diff_list"][embedding] = entity_measures[other_key]["RANK"].mrr[embedding] - entity_measures[key]["RANK"].mrr[embedding]
                
        json_output = []
        for i, key in enumerate(entity_measures.keys()):
            json_output.append(deepcopy(entity_measures[key]))
            json_output[i]["RANK"] = json_output[i]["RANK"].as_dict()

        json_output_normalized = []
        if normalization_scores is not None:
            for i, key in enumerate(entity_measures.keys()):
                entity_measures[key]["RANK"].normalize_to(normalization_scores)
            
            for i, key in enumerate(entity_measures.keys()):
                entity_n = entity_measures[key]["ENTITY_N"]
                entity_m = entity_measures[key]["ENTITY_M"]
                other_key = entity_m+";"+entity_n
                if other_key in entity_measures.keys():
                    if entity_measures[key]["FACTS"] >= self.zeta and entity_measures[other_key]["FACTS"] >= self.zeta:
                        entity_measures[key]["diff_list"] = {}
                        for embedding in embeddings:
                            entity_measures[key]["diff_list"][embedding] = entity_measures[other_key]["RANK"].mrr[embedding] - entity_measures[key]["RANK"].mrr[embedding]                  

                json_output_normalized.append(deepcopy(entity_measures[key]))
                json_output_normalized[i]["RANK"] = json_output_normalized[i]["RANK"].as_dict()

        json_output.sort(key=lambda val: val["FACTS"], reverse=True)
        results_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_9_hypothesis_3", "hypothesis_3.json")
        write_json(results_path, json_output)

        if normalization_scores is not None:
            json_output_normalized.sort(key=lambda val: val["FACTS"], reverse=True)
            results_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_9_hypothesis_3", "hypothesis_3_normalized.json")
            write_json(results_path, json_output_normalized)

    def entity_MRR_Sort(self, entity_scores, method_name):
        
        entity_scores = [x for x in entity_scores if x['NUM_FACTS'] > 12]
        sortedList = sorted(entity_scores, key=lambda d: d['MEASURE'][method_name]['MRR'], reverse=True)
        return sortedList

    def get_Top_N_Elements(self, entity_scores, n=5):
        Top5_Dict = {}
        for method_name in ["DE_TransE", "DE_SimplE", "DE_DistMult", 'TERO', 'ATISE', 'TFLEX']:
            sortedList= self.entity_MRR_Sort(entity_scores, method_name)
            Top5_Dict[method_name] = {}
            for i in range(0,n):
                dict_Name = "Number {}".format(i+1)

                Top5_Dict[method_name][dict_Name] = sortedList[i]
        
        results_path = os.path.join(self.params.base_directory, "result", self.params.dataset, "hypothesis_2", "top_5_entities.json")
        write_json(results_path, Top5_Dict)
        
        return


    def no_of_elements(self, no_of_elements_dataset, dataset):
        entities = {}
        relations = {}
        timestamps = {}

        for line in no_of_elements_dataset.values:
            if line[0] not in entities.keys():
                entities[line[0]] = 0
            if line[1] not in relations.keys():
                relations[line[1]] = 0
            if line[2] not in entities.keys():
                entities[line[2]] = 0
            if line[3] not in timestamps.keys():
                timestamps[line[3]] = 0
            if dataset in ['wikidata12k', 'yago11k']:
                if line[4] not in timestamps.keys():
                    timestamps[line[4]] = 0


            entities[line[0]] += 1
            relations[line[1]] += 1
            entities[line[2]] += 1
            timestamps[line[3]] += 1
            if dataset in ['wikidata12k', 'yago11k']:
                timestamps[line[4]] += 1
        
        entities_json = []
        relations_json = []
        timestamps_json = []

        for key in entities.keys():
            entities_json.append({"ENTITY": key, "COUNT": entities[key]})
        for key in relations.keys():
            relations_json.append({"RELATION": key, "COUNT": relations[key]})
        for key in timestamps.keys():
            if dataset in ['icews14']:
                sortable = key 
            elif dataset in ['wikidata12k']:
                if key == "####":
                    sortable = -10000
                else:
                    sortable = int(key)
            elif dataset in ['yago11k']:
                spli = key.split('-')
                if spli[0] == '':
                    sortable = [-int(spli[1].replace('#', '0'))]
                else:
                    sortable = [int(spli[0].replace('#', '0'))]
                sortable += [int(spli[-2].replace('#', '0'))]
                sortable += [int(spli[-1].replace('#', '0'))]
                sortable = tuple(sortable)
            timestamps_json.append({"TIMESTAMP": key, "COUNT": timestamps[key], "SORTABLE": sortable})

        print("entitiy count: " + str(len(entities_json)))
        print("relations count: " + str(len(relations_json)))
        print("timestamps count: " + str(len(timestamps_json)))
        
        entities_json.sort(key=lambda val: val["COUNT"], reverse=True)
        results_path = os.path.join(self.params.base_directory, "result", dataset, "no_of_elements", "full_entities.json")
        write_json(results_path, entities_json)

        relations_json.sort(key=lambda val: val["COUNT"], reverse=True)
        results_path = os.path.join(self.params.base_directory, "result", dataset, "no_of_elements", "full_relations.json")
        write_json(results_path, relations_json)

        timestamps_json.sort(key=lambda val: val["SORTABLE"], reverse=False)
        for t in timestamps_json:
            t.pop("SORTABLE")
        results_path = os.path.join(self.params.base_directory, "result", dataset, "no_of_elements", "full_timestamps.json")
        write_json(results_path, timestamps_json)

    def find_common_elements(self, entity_top_100):
        methods =['DE_TransE', 'DE_SimplE', 'DE_DistMult', 'TERO', 'ATISE', 'TFLEX']
        common_elements =entity_top_100['DE_TransE']['TOP']
        for method in methods:
            common_elements = set(common_elements).intersection(entity_top_100[method]['TOP'])
        #print(common_elements)
        print(len(common_elements))
        #print(len(entity_top_100['DE_DistMult']['TOP']))
        percentage =  len(common_elements) / len(entity_top_100['DE_TransE']['TOP'])
        print (percentage* 100)
        return

    def run(self):        
        embeddings = self.params.embeddings

        for dataset in self.params.datasets:            
            # # entities_path = os.path.join(self.params.base_directory, "result", dataset, "hypothesis_2", "entity.json")        
            # # entity_scores = read_json(entities_path)
            # # self.get_Top_5_Elements(entity_scores)

            # # entities_top100_path = os.path.join(self.params.base_directory, "result", dataset, "hypothesis_2","top_x_overlap", "entity_top_100_.json")      
            # # entities_top50_percentage_path = os.path.join(self.params.base_directory, "result", dataset, "hypothesis_2","top_x_overlap", "entity_top_50_percentage.json")     
            
            # # top = read_json(entities_top100_path)
            # # top_percentage = read_json(entities_top50_percentage_path)

            # # self.find_common_elements(top)
            # # self.find_common_elements(top_percentage)

            # relation_properties_hypothesis = RelationPropertiesHypothesis(self.params, dataset, mode="timestamps")
            # relation_properties_hypothesis.run_analysis()

            # if dataset in ['icews14']:
            #     no_of_elements_path = os.path.join(self.params.base_directory, "datasets", dataset, "full.txt")
            # else:
            #     no_of_elements_path = os.path.join(self.params.base_directory, "datasets", dataset, "triple2id.txt")
            # no_of_elements_dataset = self.read_csv(no_of_elements_path)
            # self.no_of_elements(no_of_elements_dataset, dataset)

            time_density_hypothesis = TimeDensityHypothesis(self.params, dataset)
            time_density_hypothesis.run_analysis()

            # voting_hypothesis = VotingHypothesis(self.params, dataset)
            # voting_hypothesis.run_analysis()

            for split in self.params.splits:

                ranks_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "ranked_quads.json")
                # ranked_quads = read_json(ranks_path)

                # self.calculate_overall_scores(ranked_quads, embeddings, dataset, split)

                # overall_scores_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "overall_scores.json")        
                # overall_scores = read_json(overall_scores_path)

                # self.semester_9_hypothesis_1(ranked_quads, embeddings, dataset, split)
                # self.semester_9_hypothesis_2(ranked_quads, embeddings, dataset, split, normalization_scores=overall_scores)
                # self.semester_9_hypothesis_3(ranked_quads, embeddings, dataset, split, normalization_scores=overall_scores)
                # self.semester_9_hypothesis_2_top_x(embeddings, dataset, split, top_num=10)
                # self.semester_9_hypothesis_2_top_x(embeddings, dataset, split, top_num=20)
                # self.semester_9_hypothesis_2_top_x(embeddings, dataset, split, top_num=100)
                # self.semester_9_hypothesis_2_top_x(embeddings, dataset, split, top_num=50, percentage=True)

    def average_timestamp_precision(self):
        for dataset in self.params.datasets:
            for split in self.params.splits:
                print(f"Calculating degree of error per prediction  and MAE on {dataset} ({split})")
                for embedding in self.params.embeddings:
                    # file paths
                    predictions_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "best_predictions.json")
                    avg_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "timestamp_prediction_avg.json")
                    errdist_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "time_prediction_distribution", embedding + ".dat")
                    predictions = read_json(predictions_path, self.params.verbose)

                    #get differences and average and write to files
                    self.predictions_error(predictions, predictions_path, dataset, embedding)
                    self.best_predictions_time_difference_avg(predictions, avg_path, embedding)

                    if 'BEST_DIFFERENCE' in predictions[1]['BEST_PREDICTION'][embedding].keys():
                        key = [['BEST_PREDICTION', embedding, 'BEST_DIFFERENCE'],['BEST_PREDICTION', embedding, 'WORST_DIFFERENCE']]
                    else:
                        key = [['BEST_PREDICTION', embedding, 'DIFFERENCE']]
                    
                    for k in key:
                        p = errdist_path
                        if k[2] != "DIFFERENCE":
                            p = errdist_path.replace(".dat", f"_{k[2].lower()[:4]}.dat")
                        self.count_occurences(predictions, p, k)

    def predictions_error(self, best_predictions, predictions_path, dataset, embedding):
        for i in best_predictions:

            # skip predictions for which we have no answer
            if i['ANSWER']=='-':
                    continue
            
            match(dataset):
                case 'icews14':
                    answer = date.fromisoformat(i['ANSWER'])
                    prediction = date.fromisoformat(i['BEST_PREDICTION'][embedding]['PREDICTION'])
                    difference = (prediction-answer).days
                    i['BEST_PREDICTION'][embedding]['DIFFERENCE'] = difference
                case 'wikidata12k' | 'yago11k':
                    if type(i['BEST_PREDICTION'][embedding]['PREDICTION']) is list:
                        answer = int(i['ANSWER'])
                        time_begin = i['BEST_PREDICTION'][embedding]['PREDICTION'][0]
                        time_end = i['BEST_PREDICTION'][embedding]['PREDICTION'][1]
                        difference_begin = time_begin-answer
                        difference_end = time_end-answer
                        if time_begin < answer < time_end:
                            best_case = 0
                            worst_case = 0
                        elif abs(difference_begin) < abs(difference_end):
                            best_case = difference_begin
                            worst_case = difference_end
                        else:
                            best_case = difference_end
                            worst_case = difference_begin
                        i['BEST_PREDICTION'][embedding]['BEST_DIFFERENCE'] = best_case
                        i['BEST_PREDICTION'][embedding]['WORST_DIFFERENCE'] = worst_case
                    else:
                        answer = int(i['ANSWER'])
                        prediction = int(i['BEST_PREDICTION'][embedding]['PREDICTION'])
                        difference = prediction-answer
                        i['BEST_PREDICTION'][embedding]['DIFFERENCE'] = difference

        # write to file
        write_json(predictions_path, best_predictions, self.params.verbose)
                   
    def best_predictions_time_difference_avg(self, best_predictions, avg_path, embedding):
        # load avg (to not over-write)
        avg = read_json(avg_path, self.params.verbose) if exists(avg_path) else {}

        # filter predictions w no answer
        predictions = list(filter( lambda x: 'DIFFERENCE' in x['BEST_PREDICTION'][embedding].keys(), best_predictions))

        # find avg
        if len(predictions) > 0:
            no_predictions = len(predictions)
            total_difference = sum(abs(i['BEST_PREDICTION'][embedding]['DIFFERENCE']) for i in predictions)
            avg[embedding] = total_difference/no_predictions
        else:
            predictions = list(filter( lambda x: 'BEST_DIFFERENCE' in x['BEST_PREDICTION'][embedding].keys(), best_predictions))
            no_predictions = len(predictions)
            total_best_difference = sum(abs(i['BEST_PREDICTION'][embedding]['BEST_DIFFERENCE']) for i in predictions)
            total_worst_difference = sum(abs(i['BEST_PREDICTION'][embedding]['WORST_DIFFERENCE']) for i in predictions)
            if embedding not in avg.keys():
                avg[embedding] = {}
            avg[embedding]['BEST'] = total_best_difference/no_predictions
            avg[embedding]['WORST'] = total_worst_difference/no_predictions

        # write to file
        write_json(avg_path, avg, self.params.verbose)

    def count_occurences(self, input, output_path, key):
        occurences = {}
        for i in input:
            # skip predictions for which we have no answer
            if i['ANSWER']=='-':
                    continue
            
            diff = getdctval(i, key)
            if diff not in occurences.keys():
                occurences[diff] = 1
            else:
                occurences[diff] += 1

        output=''
        for x in sorted(occurences):
            output += f"{x} {occurences[x]}\n"

        write(output_path, output)

    def dataset_timestamps(self):
        dataset = DatasetHandler(self.params, "wikidata12k")
        dataset.read_full_dataset()
        y = set()
        for x in dataset._rows:
            if x['start_timestamp'] != '-':
                y.add(int(x['start_timestamp']))
            if x['end_timestamp'] != '-':
                y.add(int(x['end_timestamp']))
        print("minimum:", min(y - {19, 25, 97, 98, 196, 229, 265, 266, 280, 285}))
        i=0
        for z in y:
            if z < 1700:
                i=i+1
        print("count:", i)
        print(max(y))
        print(len(y))