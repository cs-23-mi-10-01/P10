
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

        
    def format(self):
        embeddings = ["DE_TransE", "DE_DistMult", "DE_SimplE", "TERO", "ATISE", "TimePlex"]
        datasets = ['icews14', 'wikidata12k', 'yago11k']
        splits = ['original']
        metric = "MRR"

        static_text_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "semester_10_hypothesis_3_static_text.txt")
        shorthand_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "method_shorthand.json")

        static_text = self.read_text(static_text_path)
        shorthand = self.read_json(shorthand_path)

        for dataset in datasets:
            for split in splits:
                results_dir = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_10_hypothesis_3")
                overall_scores_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "overall_scores.json")
                overall_scores = self.read_json(overall_scores_path)

                data_string = ""
                for embedding in embeddings:
                    overall_score = overall_scores[embedding][metric]
                    data_string += f"{shorthand[embedding]} &"
                    # data_string += f"{overall_score:.2f} &"

                    for property in ["symmetric", "anti-symmetric", "inverse", "reflexive"]:
                        results_path = os.path.join(results_dir, "relation_property_" + property + "_" + self.mode + ".json")

                        if exists(results_path):
                            results = self.read_json(results_path)

                            not_in_class = results["not " + property]
                            in_class = results[property]
                            
                            if embedding in not_in_class.keys():
                                not_in_class_score = not_in_class[embedding][metric]
                                data_string += f" {not_in_class_score:.2f} &"
                            else:
                                data_string += " 0.xx &"

                            if embedding in in_class.keys():
                                in_class_score = in_class[embedding][metric]
                                data_string += f" {in_class_score:.2f}"
                                if embedding in not_in_class.keys():
                                    percentage_dif = (in_class_score / not_in_class_score) * 100.0 - 100.0
                                    data_string += r" (\textcolor{text" + f"{'green}{$+' if percentage_dif > 0 else 'red}{$-'}{abs(percentage_dif):.0f}" + r"\%$})"
                                data_string += " &"
                            else:
                                data_string += " 0.xx &"
                    
                    data_string = data_string[0:-2] + r" \\ " + "\n"

                text = static_text.replace(
                    "%1", dataset).replace(
                    "%2", data_string)                
                
                output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "semester_10_hypothesis_3", dataset+"_"+split+"_"+self.mode+".tex")
                write(output_path, text)


