import torch
import numpy as np
import time
import datetime
from scripts import remove_unwanted_symbols_from_str
from dateutil.relativedelta import relativedelta

class RankCalculator:
    def __init__(self, params, model, dataset):
        self.params = params
        self.kg = model.kg
        self.model = model
        self.dataset = dataset

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

    def get_rank(self, scores):  # assuming the first fact is the correct fact
        return torch.sum((scores < scores[0]).float()).item() + 1

    def get_ent_id(self, entity):
        return self.kg.entity_dict[remove_unwanted_symbols_from_str(entity)]

    def get_rel_id(self, relation):
        return self.kg.relation_dict[relation]
    
    def get_day_from_timestamp(self, timestamp):
        end_sec = time.mktime(time.strptime(timestamp, '%Y-%m-%d'))
        return int((end_sec - self.kg.start_sec) / (self.kg.gran * 24 * 60 * 60))

    def get_time_id_from_timestamp(self, timestamp):
        if self.dataset in ['icews14']:
            return self.get_day_from_timestamp(timestamp)
        
        if self.dataset in ['yago11k']: 
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

        return np.array(sim_facts, dtype='float64')

    def get_rank_of(self, head, relation, tail, time_from, time_to, answer):
        target = "?"
        if head == "0":
            target = "h"
        elif relation == "0":
            target = "r"
        elif tail == "0":
            target = "t"
        elif time_from == "0":
            target = "Tf"
        elif time_to == "0":
            target = "Tt"

        facts = self.simulate_facts(head, relation, tail, time_from, target, answer)
        scores = self.model.forward(facts)
        rank = self.get_rank(scores)

        return int(rank)
