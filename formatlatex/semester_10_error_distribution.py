
import os
import json
from scripts import exists, write, read_text, read_json, divide_into_buckets


class FormatErrorDistribution():
    def __init__(self, params):
        self.params = params
        self.datasets = params.datasets
        self.splits = params.splits

    def format(self):
        static_text_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "semester_10_error_distribution_text.txt")
        subfigure_text_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "semester_10_error_distribution_subfigure.txt")
        shorthand_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "method_shorthand.json")
        
        static_text = read_text(static_text_path)
        subfigure_text = read_text(subfigure_text_path)
        shorthand = read_json(shorthand_path)

        
        for dataset in self.datasets:
            for split in self.splits:
                best_predictions_path = os.path.join(self.params.base_directory, "result", dataset, "split_" + split, "best_predictions.json")

                best_predictions = read_json(best_predictions_path)

                differences = {}

                for quad in best_predictions:
                    for embedding in quad["BEST_PREDICTION"]:
                        if embedding not in differences.keys():
                            differences[embedding] = {}
                        
                        best_difference = False
                        if "DIFFERENCE" in quad["BEST_PREDICTION"][embedding].keys():
                            difference = quad["BEST_PREDICTION"][embedding]["DIFFERENCE"]
                            best_difference = True
                        if "BEST_DIFFERENCE" in quad["BEST_PREDICTION"][embedding].keys():
                            difference = quad["BEST_PREDICTION"][embedding]["BEST_DIFFERENCE"]
                            best_difference = True

                        if best_difference:
                            if "BEST" not in differences[embedding].keys():
                                differences[embedding]["BEST"] = {}
                            if difference not in differences[embedding].keys():
                                differences[embedding]["BEST"][difference] = 0
                            
                            differences[embedding]["BEST"][difference] += 1
                        else:
                            continue
                        
                        if "WORST_DIFFERENCE" in quad["BEST_PREDICTION"][embedding].keys():
                            difference = quad["BEST_PREDICTION"][embedding]["WORST_DIFFERENCE"]

                            if "WORST" not in differences[embedding].keys():
                                differences[embedding]["WORST"] = {}
                            if difference not in differences[embedding].keys():
                                differences[embedding]["WORST"][difference] = 0

                            differences[embedding]["WORST"][difference] += 1

                for embedding in differences.keys():
                    coordinate_sets = {}
                    for set in ["BEST", "WORST"]:
                        if set not in differences[embedding].keys():
                            continue

                        coordinate_sets[set] = []

                        for d in range(min(differences[embedding][set].keys()), max(differences[embedding][set].keys()) + 1):
                            if d in differences[embedding][set].keys():
                                coordinate_sets[set].append([d, differences[embedding][set][d]])
                            else:
                                coordinate_sets[set].append([d, 0])
                        
                        coordinate_sets[set] = divide_into_buckets(coordinate_sets[set], buckets=100)

                    coordinate_sets_text = {}
                    ymin = 0
                    ymax = 0

                    for set in coordinate_sets.keys():
                        first_time = coordinate_sets[set][0][0]
                        last_time = coordinate_sets[set][-1][0]

                        coordinate_sets_text[set] = f"""({first_time}, 0)""" + "\n"
                        for coordinate in coordinate_sets[set]:                    
                            coordinate_sets_text[set] += f"""({coordinate[0]}, {coordinate[1]})""" + "\n"
                            if ymax < coordinate[1] * 1.2:
                                ymax = coordinate[1] * 1.2
                        coordinate_sets_text[set] += f"""({last_time}, 0)"""
                                    
                    if "WORST" not in coordinate_sets_text.keys():
                        subfigure = subfigure_text.replace(
                            "%1", coordinate_sets_text["BEST"]).replace(
                            "%2", "ourdarkblue").replace(
                            "%3", "ourlightblue").replace(
                            "%4", "1.0")
                    else:
                        subfigure = subfigure_text.replace(
                            "%1", coordinate_sets_text["WORST"]).replace(
                            "%2", "ourdarkred").replace(
                            "%3", "ourlightred").replace(
                            "%4", "0.5") + "\n"
                        
                        subfigure += subfigure_text.replace(
                            "%1", coordinate_sets_text["BEST"]).replace(
                            "%2", "ourdarkblue").replace(
                            "%3", "ourlightblue").replace(
                            "%4", "1.0") + "\n"
                        
                        subfigure += subfigure_text.replace(
                            "%1", coordinate_sets_text["BEST"]).replace(
                            "%2", "ourdarkblue").replace(
                            "%3", "ourlightblue").replace(
                            "%4", "1.0")

                    text = static_text.replace(
                        "%1", str(ymin)).replace(
                        "%2", str(ymax)).replace(
                        "%3", subfigure).replace(
                        "%4", shorthand[embedding]).replace(
                        "%5", shorthand[dataset]).replace(
                        "%6", embedding.lower()).replace(
                        "%7", dataset)
                    
                    output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "semester_10_time_prediction_distributon", embedding.lower()+"_"+dataset+"_"+split+".tex")
                    write(output_path, text)
                    

                        



