import torch
import numpy as np
import datetime
import os
from scripts import read_json
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
        if self.dataset_handler.entity2id(entity) in self.entity_map.keys():
            return self.entity_map[self.dataset_handler.entity2id(entity)]
        else:
            bla = self.entity_map["<OOV>"]
            return bla

    def get_rel_id(self, relation):
        return self.relation_map[self.dataset_handler.relation2id(relation)]
    
    def get_time_id(self, timestamp):
        if str(timestamp) in self.year2id.keys():
            return self.year2id[str(timestamp)]
        else:
            return self.year2id["UNK-TIME"]
    
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
    
    def all_entities(self):
        return self.dataset_handler.all_entities()
    
    def all_relations(self):
        return self.dataset_handler.all_relations()
    
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
        if time_from != "0":
            year_from, month_from, day_from = self.split_timestamp(time_from)
        if time_to != "0":
            year_to, month_to, day_to = self.split_timestamp(time_to)

        match target:
            case "h":
                sim_facts = [[entity, relation, tail, year_from, year_to] for entity in range(self.all_entities())]
                sim_facts = [[answer, relation, tail, year_from, year_to]] + sim_facts
            case "r":
                sim_facts = [[head, rel, tail, year_from, year_to] for rel in range(self.all_relations())]
                sim_facts = [[head, answer, tail, year_from, year_to]] + sim_facts
            case "t":
                sim_facts = [[head, relation, entity, year_from, year_to] for entity in range(self.all_entities())]
                sim_facts = [[head, relation, answer, year_from, year_to]] + sim_facts
            case "Tf":
                sim_facts = []
                for year in self.year2id.keys():
                    if year == "UNK-TIME":
                        continue
                    sim_facts.append([head, relation, tail, year, year_to])
                ans_year, ans_month, ans_day = self.split_timestamp(answer)
                sim_facts = [[head, relation, tail, ans_year, year_to]] + sim_facts
            case "Tt":
                sim_facts = []
                for year in self.year2id.keys():
                    if year == "UNK-TIME":
                        continue
                    sim_facts.append([head, relation, tail, year_from, year])
                ans_year, ans_month, ans_day = self.split_timestamp(answer)
                sim_facts = [[head, relation, tail, year_from, ans_year]] + sim_facts
            case _:
                raise Exception("Unknown target")

        return sim_facts
    
    def facts_as_ids(self, facts):
        ret_facts = [[self.get_ent_id(f[0]),
                      self.get_rel_id(f[1]), 
                      self.get_ent_id(f[2]), 
                      self.get_time_id(f[3]), 
                      self._if_year(f[3]), 
                      self.get_time_id(f[4]), 
                      self._if_year(f[4]), 
                      self.time_str_id(f[3], f[4]), 
                      self.interval_id(f[3], f[4])] for f in facts]

        return ret_facts

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
        
        simulated_facts = self.simulate_facts(head, relation, tail, time_from, time_to, target, answer)
        facts = self.facts_as_ids(simulated_facts)
        s, r, o, t = self.shred_facts(facts)
        scores = self.model.forward(s, r, o, t)
        rank = self.get_rank(scores)

        return int(rank)
