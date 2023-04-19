
import os
import ast
import json
import datetime
from scripts import read_json, write_json, exists, date_to_iso
from statistics.measure import Measure
from dateutil.relativedelta import relativedelta

class VotingHypothesis():
    def __init__(self, params, dataset, split = "original"):
        self.params = params
        self.dataset = dataset
        self.split = split
    
    def _from_embedding_specific_format_to_iso(self, best_prediction):
        if self.dataset in ["icews14"]:
            as_list = ast.literal_eval(best_prediction)
            return f"{as_list[3]:04d}-{as_list[4]:02d}-{as_list[5]:02d}"
        
    def _first_last(self, a, b):
        return (a if a < b else b, a if a >= b else b)
    
    def _difference(self, a, b):
        sign = 1 if a < b else -1

        if self.dataset in ["icews14"]:
            delta_date = relativedelta(days=1)

        first, last = self._first_last(datetime.date.fromisoformat(a), datetime.date.fromisoformat(b))

        counter = 0
        while first < last:
            first += delta_date
            counter += 1
        return counter * sign
    
    def _move_date(self, date_iso, move_steps):
        if self.dataset in ["icews14"]:
            delta_date = relativedelta(days=move_steps)

        date = datetime.date.fromisoformat(date_iso)

        return date + delta_date

    def _average(self, dates_iso):
        aggregate = 0
        base_date_iso = "2014-01-01"
        for date_iso in dates_iso:
            aggregate += self._difference(base_date_iso, date_iso)
        aggregate = aggregate // len(dates_iso)

        return date_to_iso(self._move_date(base_date_iso, aggregate))

    def _create_mrp(self, mrp_path):
        best_time_predictions_path = os.path.join(self.params.base_directory, "result", self.dataset, "split_" + self.split, "best_predictions.json")

        best_time_predictions = read_json(best_time_predictions_path)

        measure = Measure()
        avg_measure = Measure()
        for prediction_quad in best_time_predictions:
            if "BEST_PREDICTION" in prediction_quad.keys():
                rankings = {}
                best_answers = []

                for embedding in prediction_quad["BEST_PREDICTION"]:
                    best_iso = self._from_embedding_specific_format_to_iso(prediction_quad["BEST_PREDICTION"][embedding])
                    answer_iso = prediction_quad["ANSWER"]

                    rankings[embedding] = self._difference(best_iso, answer_iso) + 1
                    best_answers.append(best_iso)
                
                avg_date = self._average(best_answers)
                avg_ranking = {"AVG": self._difference(avg_date, answer_iso) + 1}

                measure.update(rankings)
                avg_measure.update(avg_ranking)

        write_json(mrp_path, measure.as_mrp() | avg_measure.as_mrp())

    def run_analysis(self):
        mrp_path = os.path.join(self.params.base_directory, "statistics", "resources", self.dataset, "split_" + self.split, "overall_mrp_scores.json")
        

        if not exists(mrp_path):
            self._create_mrp(mrp_path)
        
        mrp = read_json(mrp_path)




