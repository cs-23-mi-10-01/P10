
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
    
    def _from_embedding_specific_format_to_iso(self, best_prediction, embedding = None):
        if self.dataset in ["icews14"]:
            return best_prediction
        if self.dataset in ["wikidata12k", "yago11k"]:
            mod_best_prediction = best_prediction
            
            if embedding in ["TERO", "ATISE"]:
                if type(mod_best_prediction) is list:
                    mod_best_prediction = (mod_best_prediction[0] + mod_best_prediction[1]) // 2
            if type(mod_best_prediction) is str:
                mod_best_prediction = int(mod_best_prediction)

            return f"{mod_best_prediction:04d}-01-01"
        
    def _first_last(self, a, b):
        return (a if a < b else b, a if a >= b else b)
    
    def _abs(self, x):
        return x if x >= 0 else -x
    
    def _difference(self, a, b):
        sign = 1 if a < b else -1

        if self.dataset in ["icews14"]:
            delta_date = relativedelta(days=1)
        if self.dataset in ["wikidata12k", "yago11k"]:
            delta_date = relativedelta(years=1)

        first, last = self._first_last(datetime.date.fromisoformat(a), datetime.date.fromisoformat(b))

        counter = 0
        while first < last:
            first += delta_date
            counter += 1
        return counter * sign
    
    def _move_date(self, date_iso, move_steps):
        if self.dataset in ["icews14"]:
            delta_date = relativedelta(days=move_steps)
        if self.dataset in ["wikidata12k", "yago11k"]:
            delta_date = relativedelta(years=move_steps)

        date = datetime.date.fromisoformat(date_iso)

        return date + delta_date

    def _average(self, dates_iso):
        aggregate = 0
        base_date_iso = "2014-01-01"
        for date_iso in dates_iso:
            aggregate += self._difference(base_date_iso, date_iso)
        average = aggregate // len(dates_iso)

        return date_to_iso(self._move_date(base_date_iso, average))

    def run_analysis(self):
        mrp_path = os.path.join(self.params.base_directory, "result", self.dataset, "split_" + self.split, "overall_precision_scores.json")
        best_time_predictions_path = os.path.join(self.params.base_directory, "result", self.dataset, "split_" + self.split, "best_predictions.json")

        best_time_predictions = read_json(best_time_predictions_path)

        measure = Measure()
        avg_measure = Measure()
        for i, prediction_quad in enumerate(best_time_predictions):
            if i % 1000 == 0:
                print("Voting on fact " + str(i) + "-" + str(i + 999) + " (total number: " + str(len(best_time_predictions)) + ") " \
                      + "on dataset " + self.dataset + ", split " + self.split)

            if "BEST_PREDICTION" in prediction_quad.keys():
                if prediction_quad["ANSWER"] == '-':
                    continue

                answer_iso = self._from_embedding_specific_format_to_iso(prediction_quad["ANSWER"])
                if answer_iso[0] == '-':
                    continue

                rankings = {}
                best_answers = []

                for embedding in prediction_quad["BEST_PREDICTION"]:
                    best_iso = self._from_embedding_specific_format_to_iso(prediction_quad["BEST_PREDICTION"][embedding]["PREDICTION"], embedding)
                    if best_iso[0] == '-':
                        continue                    

                    rankings[embedding] = self._abs(self._difference(best_iso, answer_iso))
                    best_answers.append(best_iso)
                
                avg_date = self._average(best_answers)
                avg_ranking = {"AVG": self._abs(self._difference(avg_date, answer_iso))}

                measure.update(rankings)
                avg_measure.update(avg_ranking)
            
        measure.normalize()
        avg_measure.normalize()

        write_json(mrp_path, measure.as_mrp() | avg_measure.as_mrp())




