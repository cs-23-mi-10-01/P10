
import os
import json
from scripts import exists, write, read_text, read_json


class FormatVotingHypothesis():    
    def __init__(self, params):
        self.params = params
        
    def format(self):
        embeddings = ["DE_TransE", "DE_DistMult", "DE_SimplE", "TERO", "ATISE", "TimePlex", "AVG"]
        datasets = ["icews14", "wikidata12k", "yago11k"]
        splits = ["original"]
        metric = "MRP"

        static_text_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "semester_10_voting_hypothesis_text.txt")
        shorthand_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "method_shorthand.json")

        static_text = read_text(static_text_path)
        shorthand = read_json(shorthand_path)

        data_string = ""
        for embedding in embeddings:
            if embedding == "AVG":
                data_string = data_string[0:-1] + r"\hline " + "\n"
            data_string += f"{shorthand[embedding]} & "

            for dataset in datasets:
                for split in splits:
                    results_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "overall_mrp_scores.json")
                    if not exists(results_path):
                        data_string += f"0.xx & "
                        continue
                    results = read_json(results_path)

                    if embedding in results.keys():
                        data_string += f"{results[embedding][metric]:.2f} & "
                    else:
                        data_string += f"0.xx & "
            data_string = data_string[0:-2] + r"\\ " + "\n"

        data_string = data_string[0:-1] + r"\hline " + "\n"
        text = static_text.replace(
            "%1", data_string)
                    
        output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "semester_10_voting_hypothesis.tex")
        write(output_path, text)