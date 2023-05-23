
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
        shorthand_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "method_shorthand.json")
        
        static_text = read_text(static_text_path)
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
                        
                        difference = quad["BEST_PREDICTION"][embedding]["DIFFERENCE"]

                        if difference not in differences[embedding].keys():
                            differences[embedding][difference] = 0
                        
                        differences[embedding][difference] += 1

                for embedding in differences.keys():
                    coordinates = []

                    for d in range(min(differences[embedding].keys()), max(differences[embedding].keys()) + 1):
                        if d in differences[embedding].keys():
                            coordinates.append([d, differences[embedding][d]])
                        else:
                            coordinates.append([d, 0])
                    
                    coordinates = divide_into_buckets(coordinates, buckets=100)

                    first_time = coordinates[0][0]
                    last_time = coordinates[-1][0]
                    ymin = 0
                    ymax = 0

                    coordinates_text = f"""({first_time}, 0)""" + "\n"
                    for coordinate in coordinates:                    
                        coordinates_text += f"""({coordinate[0]}, {coordinate[1]})""" + "\n"
                        if ymax < coordinate[1] * 1.2:
                            ymax = coordinate[1] * 1.2
                    coordinates_text += f"""({last_time}, 0)"""

                    text = static_text.replace(
                        "%1", str(ymin)).replace(
                        "%2", str(ymax)).replace(
                        "%3", coordinates_text).replace(
                        "%4", shorthand[embedding]).replace(
                        "%5", shorthand[dataset]).replace(
                        "%6", embedding.lower()).replace(
                        "%7", dataset)
                    
                    output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "semester_10_time_prediction_distributon", embedding.lower()+"_"+dataset+"_"+split+".tex")
                    write(output_path, text)
                    

                        



