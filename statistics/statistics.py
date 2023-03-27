
import os
import json
import pandas

from scripts import touch
from statistics.measure import Measure
from copy import deepcopy
from dataset_handler.dataset_handler import DatasetHandler


class Statistics():
    def __init__(self, params) -> None:
        self.params = params
        self.gamma = 15
        self.zeta = 5

    def write_json(self, path, dict):
        print("Writing to file " + path + "...")
        touch(path)
        out_file = open(path, "w", encoding="utf8")
        json.dump(dict, out_file, indent=4)
        out_file.close()
    
    def read_json(self, path):
        print("Reading from file " + path + "...")
        in_file = open(path, "r", encoding="utf8")
        dict = json.load(in_file)
        in_file.close()
        return dict
    
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
                if embedding == "TFLEX":
                    if not (quad["TAIL"] == "0" or quad["TIME"] == "0"):
                        continue
                
                ranks[embedding] = int(float(quad["RANK"][embedding]))
            measure.update(ranks)
        
        measure.normalize()
        measure.print()

        results_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "overall_scores.json")
        self.write_json(results_path, measure.as_dict())

    def semester_9_hypothesis_1(self, ranked_quads, embeddings, dataset, split, normalization_scores = None):
        for element_type in ["HEAD", "RELATION", "TAIL", "TIME"]:
            print("Rank of question tuples when " + str(element_type) + " is the answer element:")
            
            measure = Measure()

            for quad in ranked_quads:
                if quad[element_type] is not "0":
                    continue

                ranks = {}
                for embedding in embeddings:
                    if embedding == "TFLEX":
                        if element_type not in ["TAIL", "TIME"]:
                            continue
                    
                    ranks[embedding] = int(float(quad["RANK"][embedding]))
                measure.update(ranks)
            
            measure.print()
            measure.normalize()

            results_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_9_hypothesis_1", str(element_type).lower()+".json")
            self.write_json(results_path, measure.as_dict())

            if normalization_scores is not None:
                measure.normalize_to(normalization_scores)
                results_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_9_hypothesis_1", str(element_type).lower()+"_normalized.json")
                self.write_json(results_path, measure.as_dict())

    def semester_9_hypothesis_2(self, ranked_quads, embeddings, dataset, split, normalization_scores = None):
        for element in ["ENTITY", "RELATION", "TIME"]:
            print("Testing hypothesis 2 on " + str(element) + "s:")
            element_measures = {}
            json_output = []
            json_output_normalized = []

            if element is "ENTITY":
                target_parts = ["HEAD", "TAIL"]
            else:
                target_parts = [element]

            for target_part in target_parts:
                for quad in ranked_quads:
                    if quad[target_part] is "0":
                        continue

                    if quad[target_part] not in element_measures.keys():
                        element_measures[quad[target_part]] = Measure()
                    
                    ranks = {}
                    for embedding in embeddings:
                        if embedding == "TFLEX":
                            if not (quad["TAIL"] == "0" or quad["TIME"] == "0"):
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
            self.write_json(results_path, json_output)
            
            if normalization_scores is not None:
                results_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_9_hypothesis_2", str(element).lower()+"_normalized.json")
                self.write_json(results_path, json_output_normalized)

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
            json_input = self.read_json(input_path)

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
            self.write_json(results_path, json_top)
            
            overlap_results_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_9_hypothesis_2", 
                                        "top_x_overlap", str(element).lower()+"_top_" + str(top_num) + ("pct" if percentage else "") + "_overlap.json")
            self.write_json(overlap_results_path, json_overlap)
                
    def semester_9_hypothesis_3(self, ranked_quads, embeddings, dataset, split, normalization_scores = None):        
        entity_measures = {}
        print("Testing hypothesis 3.")

        for quad in ranked_quads:
            if quad["HEAD"] is "0" or quad["TAIL"] is "0":
                continue

            entity_n = quad["HEAD"]
            entity_m = quad["TAIL"]
            key = entity_n+";"+entity_m

            if key not in entity_measures.keys():
                entity_measures[key] = {"ENTITY_N": entity_n, "ENTITY_M": entity_m, "FACTS": 0, "RANK": Measure()}
            
            ranks = {}
            for embedding in embeddings:
                if embedding == "TFLEX":
                    if not (quad["TAIL"] == "0" or quad["TIME"] == "0"):
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
                    entity_measures[key]["DIFFERENCE"] = {}
                    for embedding in embeddings:
                        entity_measures[key]["DIFFERENCE"][embedding] = entity_measures[other_key]["RANK"].mrr[embedding] - entity_measures[key]["RANK"].mrr[embedding]
                
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
                        entity_measures[key]["DIFFERENCE"] = {}
                        for embedding in embeddings:
                            entity_measures[key]["DIFFERENCE"][embedding] = entity_measures[other_key]["RANK"].mrr[embedding] - entity_measures[key]["RANK"].mrr[embedding]                  

                json_output_normalized.append(deepcopy(entity_measures[key]))
                json_output_normalized[i]["RANK"] = json_output_normalized[i]["RANK"].as_dict()

        json_output.sort(key=lambda val: val["FACTS"], reverse=True)
        results_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_9_hypothesis_3", "hypothesis_3.json")
        self.write_json(results_path, json_output)

        if normalization_scores is not None:
            json_output_normalized.sort(key=lambda val: val["FACTS"], reverse=True)
            results_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_9_hypothesis_3", "hypothesis_3_normalized.json")
            self.write_json(results_path, json_output_normalized)
    
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
        self.write_json(results_path, Top5_Dict)
        
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

            entities[line[0]] += 1
            relations[line[1]] += 1
            entities[line[2]] += 1
            timestamps[line[3]] += 1
        
        entities_json = []
        relations_json = []
        timestamps_json = []

        for key in entities.keys():
            entities_json.append({"ENTITY": key, "COUNT": entities[key]})
        for key in relations.keys():
            relations_json.append({"RELATION": key, "COUNT": relations[key]})
        for key in timestamps.keys():
            timestamps_json.append({"TIMESTAMP": key, "COUNT": timestamps[key]})

        print("entitiy count: " + str(len(entities_json)))
        print("relations count: " + str(len(relations_json)))
        print("timestamps count: " + str(len(timestamps_json)))
        
        entities_json.sort(key=lambda val: val["COUNT"], reverse=True)
        results_path = os.path.join(self.params.base_directory, "result", dataset, "no_of_elements", "train_entities.json")
        self.write_json(results_path, entities_json)

        relations_json.sort(key=lambda val: val["COUNT"], reverse=True)
        results_path = os.path.join(self.params.base_directory, "result", dataset, "no_of_elements", "train_relations.json")
        self.write_json(results_path, relations_json)

        timestamps_json.sort(key=lambda val: val["COUNT"], reverse=True)
        results_path = os.path.join(self.params.base_directory, "result", dataset, "no_of_elements", "train_timestamps.json")
        self.write_json(results_path, timestamps_json)

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

    def relation_analysis(self, reader: DatasetHandler, dataset):

        print("Analyzing relation types...")

        for mode in ["timestamps", "no-timestamps"]:
            reader.read_full_dataset()
            relations_dict = {}

            print(mode)
            for row in reader.rows():
                if dataset in ["icews14"]:
                    row["start_timestamp"] = row["timestamp"]
                    row["end_timestamp"] = "-"
                if mode == "no-timestamps":
                    row["start_timestamp"] = "-"
                    row["end_timestamp"] = "-"
            
            i = 0
            for row in reader.rows():
                if i % 10000 == 0:
                    print("Analyzing row " + str(i) + "/" + str(len(reader.rows())))
                i += 1
                row_relation = row["relation"]

                if row_relation not in relations_dict.keys():
                    relations_dict[row_relation] = {"relation": row_relation, "total": 0, "symmetric": 0, "anti-symmetric": 0, "reflexive": 0, "inverse": {}}
                
                relations_dict[row_relation]["total"] += 1

                symmetric_relations = reader.find_in_rows(head=row["tail"], relation=row_relation, tail=row["head"], start_timestamp=row["start_timestamp"], end_timestamp=row["end_timestamp"])
                if len(symmetric_relations) > 0:
                    relations_dict[row_relation]["symmetric"] += 1
                else:
                    relations_dict[row_relation]["anti-symmetric"] += 1
                
                inverse_relations = reader.find_in_rows(head=row["tail"], relation="*", tail=row["head"], start_timestamp=row["start_timestamp"], end_timestamp=row["end_timestamp"])
                for inv_row in inverse_relations:
                    if inv_row["relation"] == row_relation:
                        continue

                    if inv_row["relation"] not in relations_dict[row_relation]["inverse"].keys():
                        relations_dict[row_relation]["inverse"][inv_row["relation"]] = 0
                    
                    relations_dict[row_relation]["inverse"][inv_row["relation"]] += 1
                
                if row["head"] == row["tail"]:
                    relations_dict[row_relation]["reflexive"] += 1
            
            relations_json = []
            for key in relations_dict.keys():
                relations_json.append(relations_dict[key])
            relations_json.sort(key=lambda val: val["total"], reverse=True)

            results_path = os.path.join(self.params.base_directory, "result", dataset, "relation_analysis", "relation_types_"+mode+".json")
            self.write_json(results_path, relations_json)

    def run(self):        
        embeddings = self.params.embeddings

        for dataset in self.params.datasets:            
            # # entities_path = os.path.join(self.params.base_directory, "result", dataset, "hypothesis_2", "entity.json")        
            # # entity_scores = self.read_json(entities_path)
            # # self.get_Top_5_Elements(entity_scores)

            # # entities_top100_path = os.path.join(self.params.base_directory, "result", dataset, "hypothesis_2","top_x_overlap", "entity_top_100_.json")      
            # # entities_top50_percentage_path = os.path.join(self.params.base_directory, "result", dataset, "hypothesis_2","top_x_overlap", "entity_top_50_percentage.json")     
            
            # # top = self.read_json(entities_top100_path)
            # # top_percentage = self.read_json(entities_top50_percentage_path)

            # # self.find_common_elements(top)
            # # self.find_common_elements(top_percentage)

            # dataset_handler = DatasetHandler(self.params, dataset)
            # self.relation_analysis(dataset_handler, dataset)

            learn_path = os.path.join(self.params.base_directory, "datasets", dataset, "format_A", "split_original", "train.txt")
            no_of_elements_dataset = self.read_csv(learn_path)
            self.no_of_elements(no_of_elements_dataset, dataset)

            for split in self.params.splits:

                ranks_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "ranked_quads.json")
                ranked_quads = self.read_json(ranks_path)

                self.calculate_overall_scores(ranked_quads, embeddings, dataset, split)

                overall_scores_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "overall_scores.json")        
                overall_scores = self.read_json(overall_scores_path)

                self.semester_9_hypothesis_1(ranked_quads, embeddings, dataset, split)
                self.semester_9_hypothesis_2(ranked_quads, embeddings, dataset, split)
                self.semester_9_hypothesis_3(ranked_quads, embeddings, dataset, split)
                self.semester_9_hypothesis_2_top_x(embeddings, dataset, split, top_num=10)
                self.semester_9_hypothesis_2_top_x(embeddings, dataset, split, top_num=20)
                self.semester_9_hypothesis_2_top_x(embeddings, dataset, split, top_num=100)
                self.semester_9_hypothesis_2_top_x(embeddings, dataset, split, top_num=50, percentage=True)
