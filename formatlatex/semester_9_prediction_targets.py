
    
import os
import json
from scripts import exists, write, read_text, read_json, divide_into_buckets


class FormatPredictionTargets():
    def __init__(self, params):
        self.params = params
        self.datasets = params.datasets
        self.splits = params.splits

    def _sort_and_filter_methods(self, embeddings):
        all_embeddings = ["DE_TransE", "DE_DistMult", "DE_SimplE", "TERO", "ATISE", "TimePlex"]
        return [e for e in all_embeddings if e in embeddings]
    
    def _get_colors(self, target):
        match(target):
            case "head":
                return "color=ourdarkblue, fill=ourlightblue"
            case "relation":
                return "color=ourdarkred, fill=ourlightred"
            case "tail":
                return "color=ourdarkyellow, fill=ourlightyellow"
            case "time_from":
                return "color=ourdarkgreen, fill=ourlightgreen"
    
    def format(self):
        embeddings = ["DE_TransE", "DE_DistMult", "DE_SimplE", "ATISE", "TERO", "TimePlex"]
        metric = "MRR"

        prefix_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "semester_9_hypothesis_1_prefix.txt")
        suffix_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "semester_9_hypothesis_1_suffix.txt")
        shorthand_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "shorthand.json")
        full_name_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "full_name.json")

        prefix_text = read_text(prefix_path)
        suffix_text = read_text(suffix_path)
        shorthand = read_json(shorthand_path)
        full_name = read_json(full_name_path)

        for dataset in self.params.datasets:
            for split in self.params.splits:
                overall_scores_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "overall_scores.json")

                overall_scores = read_json(overall_scores_path)

                for normalized in ["", "_normalized"]:
                    if normalized == "_normalized":
                        continue

                    text = ""
                    highest_score = 0
                    included_embeddings = []
                    included_tick_intervals = set()

                    for prediction_target in ["head", "relation", "tail", "time_from"]:
                        scores_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "semester_9_hypothesis_1", prediction_target + normalized + ".json")

                        scores = read_json(scores_path)

                        text += r"\addplot[" + self._get_colors(prediction_target) + r"] coordinates { %" + prediction_target + "\n"
                        i = 0
                        for embedding in embeddings:
                            if embedding not in scores.keys():
                                continue

                            score = scores[embedding][metric]
                            text += f"({i}, {score}) %{embedding}" + "\n"
                            if score > highest_score:
                                highest_score = score
                            
                            if embedding not in included_embeddings:
                                included_embeddings.append(embedding)
                            included_tick_intervals.add(i)
                            i += 1
                        text += r"} ;" + "\n"
                    
                    i = 0
                    for embedding in embeddings:
                        if embedding not in overall_scores.keys():
                            continue

                        overall_score = overall_scores[embedding][metric] if embedding in overall_scores.keys() else 0
                        text += r"\addplot[ourblack,sharp plot,update limits=false,] coordinates { %" + embedding + "\n" + \
                        f"({float(i) - 0.5}, {overall_score})" + "\n" + \
                        f"({float(i) + 0.5}, {overall_score})" + "\n" + \
                        r"} ;" + "\n"

                        if embedding not in included_embeddings:
                            included_embeddings.append(embedding)
                        included_tick_intervals.add(i)
                        i += 1

                    tick_interval_text = ",".join([str(i) for i in included_tick_intervals])
                    max_y = highest_score*1.2
                    
                    mod_prefix_text = prefix_text.replace(
                        "%1", tick_interval_text).replace(
                        "%2", f"""{",".join([shorthand[e] for e in included_embeddings])}""").replace(
                        "%3", str(max_y))
                    mod_suffix_text = suffix_text.replace(
                        "%1", f"{full_name[dataset]}, split {split}").replace(
                        "%2", f"{dataset}_{split}")

                    output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "semester_9_hypothesis_1", dataset+"_"+split+normalized+".tex")
                    write(output_path, mod_prefix_text + text + mod_suffix_text)