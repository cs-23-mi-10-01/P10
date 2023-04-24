import torch
import numpy as np
import time
import datetime
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
from scripts import year_to_iso_format, read_json
from dataset_handler.dataset_handler import DatasetHandler

class RankCalculator:
    def __init__(self, params, model, dataset_name, split = "original"):
        self.params = params
        self.model = model
        self.dataset_name = dataset_name
        self.split = split

        self.dataset_resource_folder = os.path.join(self.params.base_directory, "rank", "TimePlex", "resources", dataset_name, "split_" + split)

        self.dataset_handler = DatasetHandler(self.params, self.dataset_name)
        self.year2id = read_json(os.path.join(self.dataset_resource_folder, "date_year2id.json"))
        self.entity_map = read_json(os.path.join(self.dataset_resource_folder, "entity_map.json"))
        self.relation_map = read_json(os.path.join(self.dataset_resource_folder, "relation_map.json"))
        self.interval2id = read_json(os.path.join(self.dataset_resource_folder, "date_years2interval_id.json"))
        self.time_str2id = read_json(os.path.join(self.dataset_resource_folder, "time_str2id.json"))

    def get_rank(self, scores):
        return torch.sum((scores > scores[0]).float()).item() + 1

    def get_ent_id(self, entity):
        return self.entity_map[self.dataset_handler.entity2id(entity)]

    def get_rel_id(self, relation):
        return self.relation_map[self.dataset_handler.relation2id(relation)]
    
    def get_time_id(self, timestamp):
        return self.year2id[str(timestamp)]
    
    def interval_id(self, from_year, to_year):
        if f"({from_year}, {to_year})" in self.interval2id.keys():
            return self.interval2id[f"({from_year}, {to_year})"]
        else:
            return self.interval2id["('UNK-TIME', 'UNK-TIME')"]
    
    def _year_to_iso_format(self, year):
        modified_year = str(year)
        if modified_year == '-':
            modified_year = "####"
        return modified_year + "-##-##"
    
    def time_str_id(self, from_year, to_year):
        from_year_iso = self._year_to_iso_format(from_year)
        to_year_iso = self._year_to_iso_format(to_year)
        if f"{from_year_iso}\t{to_year_iso}" in self.time_str2id.keys():
            return self.time_str2id[f"{from_year_iso}\t{to_year_iso}"]
        else:
            return self.time_str2id["####-##-##\t####-##-##"]
    
    def number_of_ents(self):
        return len(self.entity_map) - 1
    
    def number_of_rels(self):
        return len(self.relation_map)
    
    def number_of_timestamps(self):
        return len(self.year2id) - 1
    
    def _if_year(self, year):
        if year == "UNK-TIME":
            return 0
        return year
    
    def split_timestamp(self, element):
        if self.dataset_name in ['wikidata12k', 'yago11k']:
            if element == '-':
                return "UNK-TIME", "UNK-TIME", "UNK-TIME"
            return int(element), "UNK-TIME", "UNK-TIME"
        else:
            dt = datetime.date.fromisoformat(element)
            return dt.year, dt.month, dt.day

    def simulate_facts(self, head, relation, tail, time_from, time_to, target, answer):
        if head != "0":
            head = self.get_ent_id(head)
        if relation != "0":
            relation = self.get_rel_id(relation)
        if tail != "0":
            tail = self.get_ent_id(tail)
        if time_from != "0":
            year_from, month_from, day_from = self.split_timestamp(time_from)
        if time_to != "0":
            year_to, month_to, day_to = self.split_timestamp(time_to)

        match target:
            case "h":
                sim_facts = [[i, relation, tail, self.get_time_id(year_from), self._if_year(year_from), self.get_time_id(year_to), self._if_year(year_to), self.time_str_id(year_from, year_to), self.interval_id(year_from, year_to)] for i in range(self.number_of_ents())]
                sim_facts = [[self.get_ent_id(answer), relation, tail, self.get_time_id(year_from), self._if_year(year_from), self.get_time_id(year_to), self._if_year(year_to), self.time_str_id(year_from, year_to), self.interval_id(year_from, year_to)]] + sim_facts
            case "r":
                sim_facts = [[head, i, tail, self.get_time_id(year_from), self._if_year(year_from), self.get_time_id(year_to), self._if_year(year_to), self.time_str_id(year_from, year_to), self.interval_id(year_from, year_to)] for i in range(self.number_of_rels())]
                sim_facts = [[head, self.get_rel_id(answer), tail, self.get_time_id(year_from), self._if_year(year_from), self.get_time_id(year_to), self._if_year(year_to), self.time_str_id(year_from, year_to), self.interval_id(year_from, year_to)]] + sim_facts
            case "t":
                sim_facts = [[head, relation, i, self.get_time_id(year_from), self._if_year(year_from), self.get_time_id(year_to), self._if_year(year_to), self.time_str_id(year_from, year_to), self.interval_id(year_from, year_to)] for i in range(self.number_of_ents())]
                sim_facts = [[head, relation, self.get_ent_id(answer), self.get_time_id(year_from), self._if_year(year_from), self.get_time_id(year_to), self._if_year(year_to), self.time_str_id(year_from, year_to), self.interval_id(year_from, year_to)]] + sim_facts
            case "Tf":
                sim_facts = []
                for key in self.year2id.keys():
                    if key == "UNK-TIME":
                        continue
                    sim_facts.append([head, relation, tail, self.get_time_id(key), self._if_year(key), self.get_time_id(year_to), self._if_year(year_to), self.time_str_id(key, year_to), self.interval_id(key, year_to)])
                ans_year, ans_month, ans_day = self.split_timestamp(answer)
                sim_facts = [[head, relation, tail, self.get_time_id(ans_year), self._if_year(ans_year), self.get_time_id(year_to), self._if_year(year_to), self.time_str_id(ans_year, year_to), self.interval_id(ans_year, year_to)]] + sim_facts
            case "Tt":
                sim_facts = []
                for key in self.year2id.keys():
                    if key == "UNK-TIME":
                        continue
                    sim_facts.append([head, relation, tail, self.get_time_id(year_from), self._if_year(year_from), self.get_time_id(key), self._if_year(key), self.time_str_id(year_from, key), self.interval_id(year_from, key)])
                ans_year, ans_month, ans_day = self.split_timestamp(answer)
                sim_facts = [[head, relation, tail, self.get_time_id(year_from), self._if_year(year_from), self.get_time_id(ans_year), self._if_year(ans_year), self.time_str_id(year_from, ans_year), self.interval_id(year_from, ans_year)]] + sim_facts
            case _:
                raise Exception("Unknown target")

        return sim_facts

    def shred_facts(self, facts):
        s = np.array([f[0] for f in facts], dtype='int64')
        r = np.array([f[1] for f in facts], dtype='int64')
        o = np.array([f[2] for f in facts], dtype='int64')
        t = np.array([f[3:] for f in facts], dtype='int64')
        
        s = torch.autograd.Variable(torch.from_numpy(s).unsqueeze(1), requires_grad=False)
        r = torch.autograd.Variable(torch.from_numpy(r).unsqueeze(1), requires_grad=False)
        o = torch.autograd.Variable(torch.from_numpy(o).unsqueeze(1), requires_grad=False)
        t = torch.autograd.Variable(torch.from_numpy(t).unsqueeze(1), requires_grad=False)
        
        return s, r, o, t

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
        
        facts = self.simulate_facts(head, relation, tail, time_from, time_to, target, answer)
        s, r, o, t = self.shred_facts(facts)
        scores = self.model.forward(s, r, o, t)
        rank = self.get_rank(scores)

        return int(rank)
