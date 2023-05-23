
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
        shorthand_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "shorthand.json")

        static_text = self.read_text(static_text_path)
        shorthand = self.read_json(shorthand_path)

        for dataset in datasets:
            if dataset in ['icews14']:
                properties = ["symmetric", "anti-symmetric", "inverse"]
                caption = "Comparison of MRR scores on test sets in ICEWS14. "+\
                    "The MRR scores of each test set with relation of the given types are compared to the MRR scores of test sets that does not have the given relation types. " +\
                    "Green numbers denote MRR scores that are significantly higher than the compared MRR score, and red numbers denote scores that are significantly lower."
            elif dataset in ['wikidata12k']:
                properties = ["anti-symmetric"]
                caption = "Comparison of MRR scores on test sets in WIKIDATA."
            elif dataset in ['yago11k']:
                properties = ["symmetric", "anti-symmetric"]
                caption = "Comparison of MRR scores on test sets in YAGO."

            for split in splits:
                results_dir = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_10_hypothesis_3")
                overall_scores_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "overall_scores.json")
                overall_scores = self.read_json(overall_scores_path)

                data_string = ""
                for embedding in embeddings:
                    overall_score = overall_scores[embedding][metric]
                    data_string += f"{shorthand[embedding]} &"
                    # data_string += f"{overall_score:.2f} &"

                    for property in properties:
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
                                    mrr_difference = in_class_score - not_in_class_score
                                    color_str = r" ($%1%2$)"
                                    sign = "-" if mrr_difference < 0 else "+"
                                    if in_class_score > not_in_class_score + (1.0 - not_in_class_score) * 0.1:
                                        color_str = r" (\textcolor{textgreen}{$%1%2$})"
                                    elif in_class_score < not_in_class_score * 0.9:
                                        color_str = r" (\textcolor{textred}{$%1%2$})"
                                    data_string += color_str.replace("%1", sign).replace("%2", f"{abs(mrr_difference):.2f}")
                                data_string += " &"
                            else:
                                data_string += " 0.xx &"
                    
                    data_string = data_string[0:-2] + r" \\ " + "\n"

                table_string = "".join(["|cc" for p in properties])
                column_string = ""
                for p in properties:
                    if p == "symmetric":
                        column_string += " & $T_S'$ & $T_S$"
                    if p == "anti-symmetric":
                        column_string += " & $T_A'$ & $T_A$"
                    if p == "inverse":
                        column_string += " & $T_I'$ & $T_I$"
                    if p == "reflexive":
                        column_string += " & $T_R'$ & $T_R$"

                text = static_text.replace(
                    "%1", dataset).replace(
                    "%2", table_string).replace(
                    "%3", column_string).replace(
                    "%4", data_string).replace(
                    "%5", caption)
                
                output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "semester_10_hypothesis_3", dataset+"_"+split+"_"+self.mode+".tex")
                write(output_path, text)


