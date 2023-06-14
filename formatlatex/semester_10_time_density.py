
import os
import json
from scripts import exists, write, read_text, read_json, divide_into_buckets


class FormatTimeDensity():
    def __init__(self, params):
        self.params = params
        
    def format(self):
        datasets = ['icews14', 'wikidata12k', 'yago11k']
        splits = ['original']
        static_text_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "semester_10_time_density_text.txt")
        full_name_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "full_name.json")

        static_text = read_text(static_text_path)
        full_name = read_json(full_name_path)

        for mode in ["small", "full"]:
            for dataset in datasets:
                for split in splits:
                    partition_path = os.path.join(self.params.base_directory, "statistics", "resources", dataset, "split_" + split, "partition.json")
                    no_of_facts_path = os.path.join(self.params.base_directory, "statistics", "resources", dataset, "split_" + split, "no_of_facts.json")

                    partitions = read_json(partition_path)
                    no_of_facts = read_json(no_of_facts_path)
                    dense_cutoff = no_of_facts["dense_cutoff"]
                    sparse_cutoff = no_of_facts["sparse_cutoff"]
                    
                    ymin = 0
                    ymax = 0

                    include = False
                    coordinates = []
                    append_coordinates = []
                    
                    for partition in partitions:
                        if include == False:
                            if mode == "small":
                                if dataset in ["icews14"]:
                                    continue
                                if dataset in ["yago11k", "wikidata12k"] and partition["start_date_as_float"] < 1700.0:
                                    continue
                            
                            if partition["no_of_facts"] == 0.0:
                                continue
                            else:
                                include = True
                        
                        append_coordinates.append([partition["start_date_as_float"], partition["no_of_facts"]])
                        
                        if partition["no_of_facts"] > 0:
                            coordinates += append_coordinates
                            append_coordinates = []

                    if len(coordinates) == 0:
                        continue

                    buckets = 100
                    divide_into_buckets(coordinates, buckets=buckets)
                    first_time_float = coordinates[0][0]
                    last_time_float = coordinates[-1][0]
                    interval = (last_time_float-first_time_float)/len(coordinates)

                    dense_coordinates = []
                    sparse_coordinates = []

                    coordinates_text = f"""({first_time_float}, 0)""" + "\n"
                    for i in range(len(coordinates)):
                        
                        coordinates_text += f"""({coordinates[i][0]}, {coordinates[i][1]})""" + "\n"
                        if ymax < coordinates[i][1] * 1.2:
                            ymax = coordinates[i][1] * 1.2

                        prev_y_val = coordinates[i][0] - interval / 2
                        prev_x_val = coordinates[i][0] + interval / 2
                        if i > 0:
                            prev_y_val = (coordinates[i-1][0] + coordinates[i][0]) / 2
                        if i < len(coordinates)-1:
                            prev_x_val = (coordinates[i][0] + coordinates[i+1][0]) / 2

                        if coordinates[i][1] < sparse_cutoff:
                            sparse_coordinates.append([prev_y_val, 1000])
                            sparse_coordinates.append([prev_x_val, 1000])
                        else:
                            sparse_coordinates.append([prev_y_val, -10])
                            sparse_coordinates.append([prev_x_val, -10])

                        if coordinates[i][1] > dense_cutoff:
                            dense_coordinates.append([prev_y_val, 1000])
                            dense_coordinates.append([prev_x_val, 1000])
                        else:
                            dense_coordinates.append([prev_y_val, -10])
                            dense_coordinates.append([prev_x_val, -10])

                    for i in range(len(dense_coordinates) - 2, 1, -1):
                        if dense_coordinates[i-1][1] == dense_coordinates[i][1] and \
                            dense_coordinates[i+1][1] == dense_coordinates[i][1]:
                            dense_coordinates.pop(i)
                    
                    for i in range(len(sparse_coordinates) - 2, 1, -1):
                        if sparse_coordinates[i-1][1] == sparse_coordinates[i][1] and \
                            sparse_coordinates[i+1][1] == sparse_coordinates[i][1]:
                            sparse_coordinates.pop(i)
                    
                    coordinates_text += f"""({last_time_float}, 0)"""

                    dense_text = f"""({dense_coordinates[0][0]}, -10)""" + "\n"
                    for dense_coordinate in dense_coordinates:
                        dense_text += f"""({dense_coordinate[0]}, {dense_coordinate[1]})""" + "\n"
                    dense_text += f"""({dense_coordinates[-1][0]}, -10)"""
                    
                    sparse_text = f"""({sparse_coordinates[0][0]}, -10)""" + "\n"
                    for sparse_coordinate in sparse_coordinates:
                        sparse_text += f"""({sparse_coordinate[0]}, {sparse_coordinate[1]})""" + "\n"
                    sparse_text += f"""({sparse_coordinates[-1][0]}, -10)"""

                    text = static_text.replace(
                        "%1", str(ymin)).replace(
                        "%2", str(ymax)).replace(
                        "%3", coordinates_text).replace(
                        "%4", dense_text).replace(
                        "%5", sparse_text).replace(
                        "%6", full_name[dataset]).replace(
                        "%7", dataset)

                    output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "semester_10_time_density", dataset+"_"+split+"_"+mode+".tex")
                    write(output_path, text)