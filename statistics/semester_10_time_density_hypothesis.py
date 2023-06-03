import os
from scripts import read_json, simulate_dates, exists, write_json, date_to_iso
import datetime
from dateutil.relativedelta import relativedelta
from statistics.measure import Measure


class TimeDensityHypothesis():
    def __init__(self, params, dataset, split = "original"):
        self.params = params
        self.dataset = dataset
        self.split = split

        if self.dataset in ['icews14']:
            self.start_date = datetime.date(2014, 1, 1)
            self.end_date = datetime.date(2015, 1, 1)
            self.delta_date = relativedelta(days=1)
        elif self.dataset in ['wikidata12k']:
            self.start_date = datetime.date(1, 1, 1)
            self.end_date = datetime.date(2021, 1, 1)
            self.delta_date = relativedelta(years=1)
        elif self.dataset in ['yago11k']:
            self.start_date = datetime.date(1, 1, 1)
            self.end_date = datetime.date(2845, 1, 1)
            self.delta_date = relativedelta(years=1)

        self.sparse_percentile = 0.25
        self.dense_percentile = 0.75

    def _include_quad(self, quad):
        if self.dataset in ["wikidata12k", "yago11k"]:
            if quad["TIME_FROM"] == "-":
                return False
            
            if quad["TIME_FROM"] == "0":
                if quad["ANSWER"] == "-":
                    return False
        
        return True
    
    def _compare_iso(self, x, y):
        return x <= y
    
    def _parse_date(self, quad):
        if quad["TIME_FROM"] != "0":
            date = quad["TIME_FROM"]
        else:
            date = quad["ANSWER"]            

        if self.dataset in ["wikidata12k", "yago11k"]:
            return f"""{int(date):04d}-01-01"""
        else:
            return date
    
    def _percentile_cutoffs(self, arr):
        arr.sort()
        number_of_nums = len(arr)
        sparse_index = int(number_of_nums * self.sparse_percentile)
        dense_index = int(number_of_nums * self.dense_percentile)
        return (arr[sparse_index], arr[dense_index])
        
    def _create_no_of_facts_resource(self, no_of_facts_path):
        test_quads_path = os.path.join(self.params.base_directory, "queries", self.dataset, "split_" + self.split, "test_quads.json")
        test_quads = read_json(test_quads_path)

        test_quads = [quad for quad in test_quads if self._include_quad(quad)]
        test_quads = [quad for quad in test_quads if quad["TIME_FROM"] != "0"]

        no_of_facts = []
        simulated_dates = simulate_dates(self.start_date, self.end_date, self.delta_date)
        for i in range(len(simulated_dates) - 1):
            no_of_facts.append({
                "start_date": date_to_iso(simulated_dates[i]),
                "end_date": date_to_iso(simulated_dates[i+1]),
                "no_of_facts": 0})
        
        for i, quad in enumerate(test_quads):
            if i % 10000 == 0:
                print(f"Time density hypothesis, dataset {self.dataset}, counting facts: Processing test quad {i}-{i+10000}, out of {len(test_quads)}")

            quad_start_date = self._parse_date(quad)

            for interval in no_of_facts:
                if interval["start_date"] <= quad_start_date and \
                    quad_start_date < interval["end_date"]:

                    interval["no_of_facts"] += 1
                    break

        no_of_facts_for_each_timestamp = []
        for i, quad in enumerate(test_quads):
            if i % 10000 == 0:
                print(f"Time density hypothesis, dataset {self.dataset}, processing median: Processing test quad {i}-{i+10000}, out of {len(test_quads)}")

            quad_start_date = self._parse_date(quad)

            for interval in no_of_facts:
                if interval["start_date"] <= quad_start_date and \
                    quad_start_date < interval["end_date"]:
                    
                    no_of_facts_for_each_timestamp.append(interval["no_of_facts"])
                    break

        sparse_cutoff, dense_cutoff = self._percentile_cutoffs(no_of_facts_for_each_timestamp)

        write_json(no_of_facts_path, {"sparse_cutoff": sparse_cutoff, "dense_cutoff": dense_cutoff, "no_of_facts": no_of_facts})

    def _create_time_density_partition(self, time_density_partition_path):
        no_of_facts_path = os.path.join(self.params.base_directory, 
                "statistics", "resources", self.dataset, 
                "split_" + self.split, "no_of_facts.json")
        
        if not exists(no_of_facts_path):
            self._create_no_of_facts_resource(no_of_facts_path)
        
        cutoffs_json = read_json(no_of_facts_path)

        partitions = []
        for interval in cutoffs_json["no_of_facts"]:
            if interval["no_of_facts"] < cutoffs_json["sparse_cutoff"]:
                partition = "sparse"
            elif cutoffs_json["dense_cutoff"] <= interval["no_of_facts"]:
                partition = "dense"
            else:
                partition = "none"
            partitions.append({
                "start_date": interval["start_date"],
                "end_date": interval["end_date"],
                "partition": partition,
                "no_of_facts": interval["no_of_facts"],
                "start_date_as_float": self._iso_to_float(interval["start_date"])
            })

        #Merge similar partitions
        for i in range(len(partitions) -2, -1, -1):
            if partitions[i]["partition"] == partitions[i+1]["partition"]:
                partitions[i+1]["start_date"] = partitions[i]["start_date"]
                partitions[i+1]["no_of_facts"] += partitions[i]["no_of_facts"]
                partitions.pop(i)
        
        #Remove start_date_as_float
        for partition in partitions:
            partition.pop("start_date_as_float")

        write_json(time_density_partition_path, partitions)

    def _iso_to_float(self, iso):
        split = iso.split("-")[0]
        if split == "":
            split = "-" + iso.split("-")[1]
        compare_date_iso = split + "-01-01"
        year = float(split)

        date = datetime.date.fromisoformat(iso)
        compare_date = datetime.date.fromisoformat(compare_date_iso)
        delta = date - compare_date

        return year + (delta.days / 356)


    def run_analysis(self):
        time_density_partition_path = os.path.join(self.params.base_directory, 
                "statistics", "resources", self.dataset, 
                "split_" + self.split, "partition.json")
        ranks_path = os.path.join(self.params.base_directory, "result", self.dataset, "split_" + self.split, "ranked_quads.json")
        time_density_path = os.path.join(self.params.base_directory, "result", self.dataset, "split_" + self.split, "time_density.json")

        if not exists(time_density_partition_path):
            self._create_time_density_partition(time_density_partition_path)
        
        partitions = read_json(time_density_partition_path)
        ranked_quads = read_json(ranks_path)

        ranked_quads = [quad for quad in ranked_quads if self._include_quad(quad)]

        sparse_measure = Measure()
        sparse_time_predictions = Measure()
        sparse_head_relation_tail_predictions = Measure()
        dense_measure = Measure()
        quads_in_sparse = 0
        quads_in_dense = 0
        quads_in_none = 0
        dense_time_predictions = Measure()
        dense_head_relation_tail_predictions = Measure()

        for i, quad in enumerate(ranked_quads):
            if i % 10000 == 0:
                print(f"Time density hypothesis, dataset {self.dataset}, measuring ranks: Processing ranked quad {i}-{i+10000}, out of {len(ranked_quads)}")

            if "RANK" not in quad.keys():
                continue

            quad_date = self._parse_date(quad)

            for partition in partitions:
                if partition["start_date"] <= quad_date and quad_date < partition["end_date"]:

                    if partition["partition"] == "sparse":
                        if quad["TIME_FROM"] != "0":
                            sparse_measure.update(quad["RANK"])

                        if quad["TIME_FROM"] == "0" or quad["TIME_TO"] == "0":
                            sparse_time_predictions.update(quad["RANK"])
                        else:
                            sparse_head_relation_tail_predictions.update(quad["RANK"])
                        quads_in_sparse += 1
                    
                    elif partition["partition"] == "dense":
                        if quad["TIME_FROM"] != "0":
                            dense_measure.update(quad["RANK"])

                        if quad["TIME_FROM"] == "0" or quad["TIME_TO"] == "0":
                            dense_time_predictions.update(quad["RANK"])
                        else:
                            dense_head_relation_tail_predictions.update(quad["RANK"])
                        quads_in_dense += 1
                    else:
                        quads_in_none += 1
                    break
        
        sparse_measure.normalize()
        sparse_time_predictions.normalize()
        sparse_head_relation_tail_predictions.normalize()
        dense_measure.normalize()
        
        dense_time_predictions.normalize()
        dense_head_relation_tail_predictions.normalize()

        write_json(time_density_path, {"dense": dense_measure.as_dict(),
                                       "dense_time_predictions": dense_time_predictions.as_dict(),
                                       "dense_head_relation_tail_predictions": dense_head_relation_tail_predictions.as_dict(),
                                       "sparse": sparse_measure.as_dict(),
                                       "sparse_time_predictions": sparse_time_predictions.as_dict(),
                                       "sparse_head_relation_tail_predictions": sparse_head_relation_tail_predictions.as_dict(), 
                                       "no_of_facts": {"dense": quads_in_dense, 
                                                       "sparse": quads_in_sparse, 
                                                       "none": quads_in_none}}})






