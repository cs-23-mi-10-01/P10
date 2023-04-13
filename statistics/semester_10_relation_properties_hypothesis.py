
import os
from scripts import read_json, write_json, exists
from dataset_handler.dataset_handler import DatasetHandler
from statistics.measure import Measure

class RelationPropertiesHypothesis():
    def __init__(self, params, dataset, split = "original", mode = "timestamps"):
        self.params = params
        self.dataset = dataset
        self.split = split
        self.mode = mode # either "timestamps" or "no-timestamps"
        
        self.dataset_handler = DatasetHandler(self.params, dataset)

        self.symmetry_threshold = 0.8
        self.anti_symmetry_threshold = 1.0
        self.inverse_threshold = 0.8
        self.reflexive_threshold = 0.8

    def _include_ranked_quad(self, quad):
        if quad["RELATION"] == "0":
            return False
        
        return True

    def _create_relation_analysis(self, relation_analysis_path):
        print("Analyzing relation types...")

        self.dataset_handler.read_full_dataset()
        relations_dict = {}

        print(self.mode)
        for row in self.dataset_handler.rows():
            if self.dataset in ["icews14"]:
                row["start_timestamp"] = row["timestamp"]
                row["end_timestamp"] = "-"
            if self.mode == "no-timestamps":
                row["start_timestamp"] = "-"
                row["end_timestamp"] = "-"
        
        i = 0
        for row in self.dataset_handler.rows():
            if i % 10000 == 0:
                print("Analyzing row " + str(i) + "/" + str(len(self.dataset_handler.rows())))
            i += 1
            row_relation = row["relation"]

            if row_relation not in relations_dict.keys():
                relations_dict[row_relation] = {"relation": row_relation, "total": 0, "symmetric": 0, "anti-symmetric": 0, "reflexive": 0, "inverse": {}}
            
            relations_dict[row_relation]["total"] += 1

            symmetric_relations = self.dataset_handler.find_in_rows(head=row["tail"], relation=row_relation, tail=row["head"], start_timestamp=row["start_timestamp"], end_timestamp=row["end_timestamp"])
            if len(symmetric_relations) > 0:
                relations_dict[row_relation]["symmetric"] += 1
            else:
                relations_dict[row_relation]["anti-symmetric"] += 1
            
            inverse_relations = self.dataset_handler.find_in_rows(head=row["tail"], relation="*", tail=row["head"], start_timestamp=row["start_timestamp"], end_timestamp=row["end_timestamp"])
            for inv_row in inverse_relations:
                if inv_row["relation"] == row_relation:
                    continue

                if inv_row["relation"] not in relations_dict[row_relation]["inverse"].keys():
                    relations_dict[row_relation]["inverse"][inv_row["relation"]] = 0
                
                relations_dict[row_relation]["inverse"][inv_row["relation"]] += 1
            
            if row["head"] == row["tail"]:
                relations_dict[row_relation]["reflexive"] += 1
        
        relations = []
        for key in relations_dict.keys():
            relations.append(relations_dict[key])
        relations.sort(key=lambda val: val["total"], reverse=True)

        relations_json = {}
        for rel in relations:
            relations_json[rel["relation"]] = rel
            relations_json[rel["relation"]].pop("relation")

        write_json(relation_analysis_path, relations_json)

    def _create_relation_classification(self, relation_classification_path):
        relation_analysis_path = os.path.join(self.params.base_directory, "statistics", "resources", self.dataset, "relation_types_" + self.mode + ".json")

        if not exists(relation_analysis_path):
            self._create_relation_analysis(relation_analysis_path)
        
        relation_analysis = read_json(relation_analysis_path)

        relation_classification = {}
        
        for relation_name in relation_analysis.keys():
            relation = relation_analysis[relation_name]
            classification = {}

            classification["symmetric"] = (relation["symmetric"] / relation["total"]) >= self.symmetry_threshold
            classification["anti-symmetric"] = (relation["anti-symmetric"] / relation["total"]) >= self.anti_symmetry_threshold
            max_inverse = 0
            for inv_name in relation["inverse"].keys():
                if relation["inverse"][inv_name] > max_inverse:
                    max_inverse = relation["inverse"][inv_name]
            classification["inverse"] = (max_inverse / relation["total"]) >= self.inverse_threshold
            classification["reflexive"] = (relation["reflexive"] / relation["total"]) >= self.reflexive_threshold

            relation_classification[relation_name] = classification

        write_json(relation_classification_path, relation_classification)

    def _analyze_with_property(self, relation_classification, ranked_quads, property = "symmetric"):
        relation_property_analysis_path = os.path.join(self.params.base_directory, "result", self.dataset, "split_" + self.split, "semester_10_hypothesis_3", "relation_property_"+property+"_"+self.mode+".json")

        measure_in_class = Measure()
        measure_no_class = Measure()

        for i, quad in enumerate(ranked_quads):
            if i % 10000 == 0:
                print(f"Relation properties hypothesis, dataset {self.dataset}, measuring classified relations: Processing ranked quad {i}-{i+10000}, out of {len(ranked_quads)}")

            if "RANK" not in quad.keys():
                continue

            if relation_classification[quad["RELATION"]][property] == True:
                measure_in_class.update(quad["RANK"])
            else:
                measure_no_class.update(quad["RANK"])

        measure_in_class.normalize()
        measure_no_class.normalize()

        write_json(relation_property_analysis_path, {property: measure_in_class.as_dict(), "not " + property: measure_no_class.as_dict()})

    def run_analysis(self):
        relation_classification_path = os.path.join(self.params.base_directory, "statistics", "resources", self.dataset, "relation_classification_" + self.mode + ".json")
        ranks_path = os.path.join(self.params.base_directory, "result", self.dataset, "split_" + self.split, "ranked_quads.json")

        if not exists(relation_classification_path):
            self._create_relation_classification(relation_classification_path)
        
        relation_classification = read_json(relation_classification_path)
        ranked_quads = read_json(ranks_path)

        ranked_quads = [quad for quad in ranked_quads if self._include_ranked_quad(quad)]

        self._analyze_with_property(relation_classification, ranked_quads, "symmetric")
        self._analyze_with_property(relation_classification, ranked_quads, "anti-symmetric")
        self._analyze_with_property(relation_classification, ranked_quads, "inverse")
        self._analyze_with_property(relation_classification, ranked_quads, "reflexive")




