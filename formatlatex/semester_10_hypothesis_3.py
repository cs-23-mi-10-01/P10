
import os
import json
from scripts import exists, write


class FormatRelationPropertyHypothesis():    
    def __init__(self, params, mode = "timestamps"):
        self.params = params
        self.mode = mode
        
    
    def read_json(self, path):
        in_file = open(path, "r", encoding="utf8")
        dict = json.load(in_file)
        in_file.close()
        return dict
    
    def read_text(self, path):
        in_file = open(path, "r", encoding="utf8")
        text = in_file.read()
        in_file.close()
        return text

        
    def format_semester_9_hypothesis_1(self):
        embeddings = ["DE_TransE", "DE_DistMult", "DE_SimplE", "TERO", "ATISE", "TimePlex"]
        metric = "MRR"

        static_text_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "semester_10_hypothesis_3_static_text.txt")
        shorthand_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "method_shorthand.json")

        static_text = self.read_text(static_text_path)
        shorthand = self.read_json(shorthand_path)

        for dataset in self.params.datasets:
            for split in self.params.splits:
                results_dir = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_10_hypothesis_3")

                data_string = ""
                for embedding in embeddings:
                    data_string += f"{shorthand[embedding]} &"

                    for property in ["symmetric", "anti-symmetric", "inverse", "reflexive"]:
                        results_path = os.path.join(results_dir, "relation_property_" + property + "_" + self.mode + ".json")

                        if exists(results_path):
                            results = self.read_json(results_path)

                            for p in [property, "not "+ property]:
                                in_class = results[p]
                                if embedding in in_class.keys():
                                    data_string += f" {in_class[embedding][metric]:.2f} &"
                                else:
                                    data_string += " 0.xx &"
                    
                    data_string = data_string[0:-2] + r" \\ " + "\n"

                text = static_text.replace(
                    "%1", dataset).replace(
                    "%2", data_string)                
                
                output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "semester_10_hypothesis_3", dataset+"_"+split+"_"+self.mode+".tex")
                write(output_path, text)


