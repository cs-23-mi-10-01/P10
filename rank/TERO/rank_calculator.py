import torch
import numpy as np
import time
import datetime
from datetime import date
from scripts import remove_unwanted_symbols_from_str
from dateutil.relativedelta import relativedelta

class RankCalculator:
    def __init__(self, params, model, dataset, embedding, mode = "rank"):
        self.params = params
        self.kg = model.kg
        self.model = model
        self.dataset = dataset
        self.embedding = embedding
        self.mode = mode

        if self.dataset in ['icews14']:
            self.start_sim_date = datetime.date(2014, 1, 1)
            self.end_sim_date = datetime.date(2015, 1, 1)
            self.delta_sim_date = datetime.timedelta(days=1)
        elif self.dataset in ['wikidata12k']:
            self.start_sim_date = datetime.date(1, 1, 1)
            self.end_sim_date = datetime.date(2021, 1, 1)
            self.delta_sim_date = relativedelta(years=1)
        elif self.dataset in ['yago11k']:
            self.start_sim_date = datetime.date(1, 1, 1)
            self.end_sim_date = datetime.date(2845, 1, 1)
            self.delta_sim_date = relativedelta(years=1)

    def get_rank(self, scores, score_of_expected=None):  # assuming the first fact is the correct fact
        score_of_expected = scores[0]

        rank = 0.0
        for score in scores:
            if score < score_of_expected:
                rank += 1.0
            elif score == score_of_expected:
                rank += 0.5
        return max([int(rank), 1])

    def get_ent_id(self, entity):
        return self.kg.entity_dict[remove_unwanted_symbols_from_str(entity)]
    
    def get_ent_from_id(self, id):
        if not hasattr(self, 'id2entity_dict'):
            self.id2entity_dict = {}
            for key in self.kg.entity_dict.keys():
                self.id2entity_dict[self.kg.entity_dict[key]] = key

        return self.id2entity_dict[id]

    def get_rel_id(self, relation):
        return self.kg.relation_dict[relation]
    
    def get_rel_from_id(self, id):
        if not hasattr(self, 'id2relation_dict'):
            self.id2relation_dict = {}
            for key in self.kg.relation_dict.keys():
                self.id2relation_dict[self.kg.relation_dict[key]] = key

        return self.id2relation_dict[id]
    
    def get_day_from_timestamp(self, timestamp):
        end_sec = time.mktime(time.strptime(timestamp, '%Y-%m-%d'))
        return int((end_sec - self.kg.start_sec) / (self.kg.gran * 24 * 60 * 60))

    def get_time_id_from_timestamp(self, timestamp):
        if self.dataset in ['icews14']:
            return self.get_day_from_timestamp(timestamp)

        if self.dataset in ['yago11k', 'wikidata12k']: 
            check_timestamp = timestamp
            if check_timestamp == '-':
                check_timestamp = "####"

            check_timestamp = check_timestamp + "-##-##"

            if check_timestamp.split('-')[0] == '####':
                return 0
            elif check_timestamp[0] == '-':
                start=-int(check_timestamp.split('-')[1].replace('#', '0'))
            elif check_timestamp[0] != '-':
                start = int(check_timestamp.split('-')[0].replace('#','0'))

            for key, time_idx in sorted(self.kg.year2id.items(), key=lambda x:x[1]):
                if start>=key[0] and start<=key[1]:
                    return time_idx
            
            return None
        
        return None
    
    def get_timestamp_from_day(self, day):
        return (self.start_sim_date + datetime.timedelta(days = int(day))).isoformat()

    def cache_timestamps(self):
        if not hasattr(self, "id2year_interval"):
            self.id2year_interval = {}

            for key in self.kg.year2id.keys():
                self.id2year_interval[self.kg.year2id[key]] = key

    def get_timestamp_from_time_id(self, time_id):
        if self.dataset in ['icews14']:
            return self.get_timestamp_from_day(time_id)
        if self.dataset in ['wikidata12k', 'yago11k']:
            self.cache_timestamps()

            year_from, year_to = self.id2year_interval[time_id]
            return (year_from, year_to)
        return None

    def simulate_facts(self, head, relation, tail, timestamp, target, answer):
        if head != "0":
            head = self.get_ent_id(head)
        if relation != "0":
            relation = self.get_rel_id(relation)
        if tail != "0":
            tail = self.get_ent_id(tail)
        if timestamp != "0":
            day = self.get_time_id_from_timestamp(timestamp)
            if day is None:
                raise Exception("None timestamp not allowed")

        match target:
            case "h":
                sim_facts = [[i, tail, relation, day] for i in range(self.kg.n_entity)]
                sim_facts = [[self.get_ent_id(answer), tail, relation, day]] + sim_facts
            case "r":
                sim_facts = [[head, tail, i, day] for i in range(self.kg.n_relation)]
                sim_facts = [[head, tail, self.get_rel_id(answer), day]] + sim_facts
            case "t":
                sim_facts = [[head, i, relation, day] for i in range(self.kg.n_entity)]
                sim_facts = [[head, self.get_ent_id(answer), relation, day]] + sim_facts
            case "Tf":
                sim_facts = []
                sim_date = self.start_sim_date
                while sim_date != self.end_sim_date:
                    year = sim_date.year
                    month = sim_date.month
                    day = sim_date.day

                    sim_timestamp = str(year) + '-' + str(month) + '-' + str(day)
                    time_id = self.get_time_id_from_timestamp(sim_timestamp)
                    if time_id is not None:
                        sim_facts.append([head, tail, relation, self.get_time_id_from_timestamp(sim_timestamp)])

                    sim_date = sim_date + self.delta_sim_date

                time_id = self.get_time_id_from_timestamp(answer)
                if time_id is None:
                    raise Exception("None timestamp not allowed")
                sim_facts = [[head, tail, relation, time_id]] + sim_facts
            case _:
                raise Exception("Unknown target")

        return sim_facts
    
    def _construct_fact_scores(self, head, relation, tail, time_from, time_to, facts, scores, target):
        fact_scores = {}

        for fact, score in zip(facts, scores):
            match target:
                case "h":
                    fact_scores[(self.get_ent_from_id(int(fact[0])), relation, tail, time_from, time_to)] = score
                case "r":
                    if int(fact[2]) < self.kg.n_relation / 2:
                        fact_scores[(head, self.get_rel_from_id(int(fact[2])), tail, time_from, time_to)] = score
                case "t":
                    fact_scores[(head, relation, self.get_ent_from_id(int(fact[1])), time_from, time_to)] = score
                case "Tf":
                    if self.embedding in ["ATISE"] and self.dataset in ["icews14"]:
                        # Account for a granularity of 3
                        modified_id = int(fact[3]) * 3
                        for i in [0, 1, 2]:
                            if modified_id + i < self.kg.n_time:
                                fact_scores[(head, relation, tail, self.get_timestamp_from_time_id(modified_id + i), time_to)] = score
                    elif self.dataset in ["wikidata12k", "yago11k"]:
                        year_from, year_to = self.get_timestamp_from_time_id(int(fact[3]))
                        for year in range(year_from, year_to):
                            fact_scores[(head, relation, tail, str(year), time_to)] = score
                    else:
                        fact_scores[(head, relation, tail, self.get_timestamp_from_time_id(int(fact[3])), time_to)] = score
        
        return fact_scores
    
    def simulate_fact_scores(self, head, relation, tail, time_from, time_to, answer):
        target = "?"
        if head == "0":
            target = "h"
        elif relation == "0":
            target = "r"
        elif tail == "0":
            target = "t"
        elif time_from == "0":
            target = "Tf"

        facts = np.array(self.simulate_facts(head, relation, tail, time_from, target, answer), dtype='float64')
        sim_scores = self.model.forward(facts).cpu().data.numpy()
        facts = facts.tolist()
        sim_scores = sim_scores.tolist()
        fact_scores = self._construct_fact_scores(head, relation, tail, time_from, time_to, facts, sim_scores, target)

        return fact_scores

    def rank_of_correct_prediction(self, fact_scores, correct_fact):
        return self.get_rank(list(fact_scores.values()))

    def best_prediction(self, fact_scores):
        highest_scoring_fact = max(fact_scores, key = lambda pair: pair[1])
        fact = highest_scoring_fact[0]
        return fact[3]