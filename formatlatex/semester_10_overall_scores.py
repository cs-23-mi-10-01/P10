
    
import os
import json
from scripts import write, read_text, read_json


class FormatOverallScores():
    def __init__(self, params):
        self.params = params
        self.datasets = params.datasets
        self.splits = params.splits

    def _round(self, val):
        return float(f"{val:.2f}")
    
    def format(self):
        embeddings = ["ensemble_naive_voting", "ensemble_decision_tree","ablation_overall", "ablation_property", "ablation_false_property", "ablation_time_density", "ablation_target" ,"ablation_no_property", "ablation_one_forth_property", "ablation_only_property", "ablation_only_target", "ablation_only_overall", "ablation_only_time_density"]
        metric = "MRR"

        prefix_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "semester_10_overall_scores_text.txt")
        shorthand_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "shorthand.json")
        full_name_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "full_name.json")

        static_text = read_text(prefix_path)
        shorthand = read_json(shorthand_path)
        full_name = read_json(full_name_path)

        for split in self.params.splits:
            rows = []

            for embedding in embeddings:
                row = [shorthand[embedding]]                

                for dataset in self.params.datasets:
                    overall_scores_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "overall_scores.json")
                    overall_scores = read_json(overall_scores_path)

                    if embedding not in overall_scores.keys():
                        row.append(None)
                        continue

                    row.append(overall_scores[embedding][metric])

                rows.append(row)

            highest_vals = [[] for _ in rows]
            for y in range(len(rows[0]) - 1):
                column = [row[y + 1] for row in rows]

                highest_val_in_column = max([self._round(e) for e in column])
                for x, element in enumerate(column):
                    if self._round(element) == highest_val_in_column:
                        highest_vals[x].append(y + 1)

            for x, row in enumerate(rows):
                y = 1
                for element in row[1:]:
                    if y in highest_vals[x]:
                        rows[x][y] = r"\textbf{" + f"{element:.2f}" + r"}"
                    else:
                        rows[x][y] = f"{element:.2f}"
                    y += 1
            
            rows_text = ""
            for x, row in enumerate(rows):
                rows_text += row[0] + " & "
                rows_text += " & ".join(row[1:]) + r" \\" + "\n"
                
            text = static_text.replace(
                "_content", rows_text)

            output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "semester_10_overall_scores", "split_"+split+".tex")
            write(output_path, text)