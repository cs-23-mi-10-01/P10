
import os
import json
from scripts import exists, write, read_text, read_json


class FormatTimeDensity():
    def __init__(self, params):
        self.params = params
        
    def format(self):
        datasets = ['icews14', 'wikidata12k', 'yago11k']
        splits = ['original']
        static_text_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "semester_10_time_density_text.txt")
        shorthand_path = os.path.join(self.params.base_directory, "formatlatex", "resources", "method_shorthand.json")

        static_text = read_text(static_text_path)
        shorthand = read_json(shorthand_path)

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

                first_time_float = coordinates[0][0]
                last_time_float = coordinates[len(coordinates) - 1][0]
                buckets = 100
                if buckets != -1:
                    interval = (last_time_float - first_time_float) / (buckets + 1)
                    for i in range(buckets):
                        coordinate_indexes_in_bucket = []
                        for j in range(len(coordinates)):
                            if coordinates[j][0] >= first_time_float + interval*i and \
                                coordinates[j][0] < first_time_float + interval*(i+1):
                                coordinate_indexes_in_bucket.append(j)

                        no_of_facts_in_bucket = 0
                        total_date_intervals_in_bucket = 0.0
                        for index in coordinate_indexes_in_bucket:
                            total_date_intervals_in_bucket += coordinates[index][0]
                            no_of_facts_in_bucket += coordinates[index][1]
                        average_date_interval = total_date_intervals_in_bucket / len(coordinate_indexes_in_bucket)
                        average_no_of_facts = no_of_facts_in_bucket / len(coordinate_indexes_in_bucket)
                        average_coord = [average_date_interval, average_no_of_facts]

                        for j in range(len(coordinate_indexes_in_bucket) -1, 0, -1):
                            coordinates.pop(coordinate_indexes_in_bucket[j])
                        coordinates[coordinate_indexes_in_bucket[0]] = average_coord
                else:
                    interval = coordinates[1][0] - coordinates[0][0]

                dense_coordinates = []
                sparse_coordinates = []

                coordinates_text = f"""({first_time_float}, 0)""" + "\n"
                for coordinate in coordinates:
                    
                    coordinates_text += f"""({coordinate[0]}, {coordinate[1]})""" + "\n"
                    if ymax < coordinate[1] * 1.2:
                        ymax = coordinate[1] * 1.2

                    if coordinate[1] < sparse_cutoff:
                        sparse_coordinates.append([coordinate[0] - interval / 2, 1000])
                        sparse_coordinates.append([coordinate[0] + interval / 2, 1000])
                    else:
                        sparse_coordinates.append([coordinate[0] - interval / 2, -10])
                        sparse_coordinates.append([coordinate[0] + interval / 2, -10])

                    if coordinate[1] > dense_cutoff:
                        dense_coordinates.append([coordinate[0] - interval / 2, 1000])
                        dense_coordinates.append([coordinate[0] + interval / 2, 1000])
                    else:
                        dense_coordinates.append([coordinate[0] - interval / 2, -10])
                        dense_coordinates.append([coordinate[0] + interval / 2, -10])

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
                    "%6", shorthand[dataset]).replace(
                    "%7", dataset)

                output_path = os.path.join(self.params.base_directory, "formatlatex", "result", "semester_10_time_density", dataset+"_"+split+".tex")
                write(output_path, text)