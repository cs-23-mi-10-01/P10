import os
from scripts import read_json, simulate_dates, exists, write_json
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

    def _include_quad(self, quad):
        if self.dataset in ["wikidata12k", "yago11k"]:
            if quad["TIME_FROM"] == "-":
                return False
        
        return True

    def _include_ranked_quad(self, quad):
        if quad["TIME_FROM"] == "0":
            return False
        
        return self._include_quad(quad)
    
    def _date_to_iso(self, date):
        return f"{date.year:04d}-{date.month:02d}-{date.day:02d}"
    
    def _compare_iso(self, x, y):
        return x <= y
    
    def _parse_start_date(self, date):
        if self.dataset in ["wikidata12k", "yago11k"]:
            return f"""{int(date):04d}-01-01"""
        else:
            return date
    
    def _median(self, arr):
        arr.sort()
        number_of_nums = len(arr)
        middle_index = number_of_nums // 2
        if number_of_nums % 2 == 0:
            return (arr[middle_index - 1] + arr[middle_index]) / 2
        else:
            return arr[middle_index]  
        
    def _create_no_of_facts_resource(self, no_of_facts_path):
        test_quads_path = os.path.join(self.params.base_directory, "queries", self.dataset, "split_" + self.split, "test_quads.json")
        test_quads = read_json(test_quads_path)

        test_quads = [quad for quad in test_quads if self._include_quad(quad)]

        no_of_facts = []
        simulated_dates = simulate_dates(self.start_date, self.end_date, self.delta_date)
        for i in range(len(simulated_dates) - 1):
            no_of_facts.append({
                "start_date": self._date_to_iso(simulated_dates[i]),
                "end_date": self._date_to_iso(simulated_dates[i+1]),
                "no_of_facts": 0})
        
        for i, quad in enumerate(test_quads):
            if i % 10000 == 0:
                print(f"Time density hypothesis, dataset {self.dataset}, counting facts: Processing test quad {i}-{i+10000}, out of {len(test_quads)}")

            quad_start_date = self._parse_start_date(quad["TIME_FROM"])

            for interval in no_of_facts:
                if interval["start_date"] <= quad_start_date and \
                    quad_start_date < interval["end_date"]:

                    interval["no_of_facts"] += 1
                    break

        no_of_facts_for_each_timestamp = []
        for i, quad in enumerate(test_quads):
            if i % 10000 == 0:
                print(f"Time density hypothesis, dataset {self.dataset}, processing median: Processing test quad {i}-{i+10000}, out of {len(test_quads)}")

            quad_start_date = self._parse_start_date(quad["TIME_FROM"])

            for interval in no_of_facts:
                if interval["start_date"] <= quad_start_date and \
                    quad_start_date < interval["end_date"]:
                    
                    no_of_facts_for_each_timestamp.append(interval["no_of_facts"])
                    break

        median = self._median(no_of_facts_for_each_timestamp)

        write_json(no_of_facts_path, {"median": median, "no_of_facts": no_of_facts})

    def _create_time_density_partition(self, time_density_partition_path):
        no_of_facts_path = os.path.join(self.params.base_directory, 
                "statistics", "resources", self.dataset, 
                "split_" + self.split, "no_of_facts.json")
        
        if not exists(no_of_facts_path):
            self._create_no_of_facts_resource(no_of_facts_path)
        
        median_json = read_json(no_of_facts_path)

        partitions = []
        for interval in median_json["no_of_facts"]:
            partitions.append({
                "start_date": interval["start_date"],
                "end_date": interval["end_date"],
                "partition": "dense" if interval["no_of_facts"] >= median_json["median"] else "sparse",
                "no_of_facts": interval["no_of_facts"]
            })

        for i in range(len(partitions) -2, -1, -1):
            if partitions[i]["partition"] == partitions[i+1]["partition"]:
                partitions[i+1]["start_date"] = partitions[i]["start_date"]
                partitions[i+1]["no_of_facts"] += partitions[i]["no_of_facts"]
                partitions.pop(i)

        write_json(time_density_partition_path, partitions)

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

        ranked_quads = [quad for quad in ranked_quads if self._include_ranked_quad(quad)]

        sparse_measure = Measure()
        dense_measure = Measure()

        for i, quad in enumerate(ranked_quads):
            if i % 10000 == 0:
                print(f"Time density hypothesis, dataset {self.dataset}, measuring ranks: Processing ranked quad {i}-{i+10000}, out of {len(ranked_quads)}")

            if "RANK" not in quad.keys():
                continue

            quad_start_date = self._parse_start_date(quad["TIME_FROM"])

            for partition in partitions:
                if partition["start_date"] <= quad_start_date and \
                    quad_start_date < partition["end_date"]:
                    
                    if partition["partition"] == "sparse":
                        sparse_measure.update(quad["RANK"])
                    if partition["partition"] == "dense":
                        dense_measure.update(quad["RANK"])
                    break
        
        sparse_measure.normalize()
        dense_measure.normalize()

        write_json(time_density_path, {"dense": dense_measure.as_dict(), "sparse": sparse_measure.as_dict()})





