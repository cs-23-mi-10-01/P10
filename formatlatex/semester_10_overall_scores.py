
    
import os
import json
from scripts import write, read_text, read_json


class FormatOverallScores():
    def __init__(self, params):
        self.params = params
        self.datasets = params.datasets
        self.splits = params.splits
    
    def format(self):
        embeddings = ["DE_TransE", "DE_DistMult", "DE_SimplE", "ATISE", "TERO", "TimePlex", "ensemble_naive_voting", "ensemble_decision_tree"]
        metric = "MRR"

        prefix_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "semester_10_overall_scores_text.txt")
        shorthand_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "shorthand.json")
        full_name_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "full_name.json")

        static_text = read_text(prefix_path)
        shorthand = read_json(shorthand_path)
        full_name = read_json(full_name_path)
        
        for split in self.params.splits:
            rows_text = ""

            for embedding in embeddings:
                row_text_elements = [shorthand[embedding]]

                for dataset in self.params.datasets:
                        overall_scores_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "overall_scores.json")
                        overall_scores = read_json(overall_scores_path)

                        if embedding not in overall_scores.keys():
                            row_text_elements.append("--")
                            continue

                        row_text_elements.append(f"{overall_scores[embedding][metric]:.2f}")

                rows_text += " & ".join(row_text_elements) + r" \\" + "\n"
                
            text = static_text.replace(
                "_content", rows_text)

            output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "semester_10_overall_scores", "split_"+split+".tex")
            write(output_path, text)