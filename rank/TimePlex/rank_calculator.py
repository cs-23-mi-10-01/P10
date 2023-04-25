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

    def get_rank(self, scores, score_of_expected):
        return torch.sum((scores > score_of_expected).float()).item() + 1

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
        return [self.dataset_handler.id2entity(e) for e in self.entity_map.keys() if e != "<OOV>"]
    
    def all_relations(self):
        return [self.dataset_handler.id2relation(r) for r in self.relation_map.keys()]
    
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

    def simulate_fact(self, head, relation, tail, time_from, time_to, target, answer):
        if time_from != "0":
            year_from, month_from, day_from = self.split_timestamp(time_from)
        if time_to != "0":
            year_to, month_to, day_to = self.split_timestamp(time_to)

        match target:
            case "h":
                sim_fact = [answer, relation, tail, year_from, year_to]
            case "r":
                sim_fact = [head, answer, tail, year_from, year_to]
            case "t":
                sim_fact = [head, relation, answer, year_from, year_to]
            case "Tf":
                ans_year, ans_month, ans_day = self.split_timestamp(answer)
                sim_fact = [head, relation, tail, ans_year, year_to]
            case "Tt":
                ans_year, ans_month, ans_day = self.split_timestamp(answer)
                sim_fact = [head, relation, tail, year_from, ans_year]
            case _:
                raise Exception("Unknown target")

        return sim_fact
    
    def fact_as_ids(self, fact):
        ret_fact = [self.get_ent_id(fact[0]),
                      self.get_rel_id(fact[1]), 
                      self.get_ent_id(fact[2]), 
                      self.get_time_id(fact[3]), 
                      self._if_year(fact[3]), 
                      self.get_time_id(fact[4]), 
                      self._if_year(fact[4]), 
                      self.time_str_id(fact[3], fact[4]), 
                      self.interval_id(fact[3], fact[4])]

        return ret_fact

    def shred_fact(self, fact):

        s = np.array([fact[0]], dtype='int64')
        r = np.array([fact[1]], dtype='int64')
        o = np.array([fact[2]], dtype='int64')
        t = np.array([fact[3:]], dtype='int64')
        
        s = torch.autograd.Variable(torch.from_numpy(s).unsqueeze(1), requires_grad=False)
        r = torch.autograd.Variable(torch.from_numpy(r).unsqueeze(1), requires_grad=False)
        o = torch.autograd.Variable(torch.from_numpy(o).unsqueeze(1), requires_grad=False)
        t = torch.autograd.Variable(torch.from_numpy(t).unsqueeze(1), requires_grad=False)
        
        return s, r, o, t

    def get_scores(self, target, s, r, o, t):
        match target:
            case "h":
                scores = self.model(None, r, o, t).data
                score_of_expected = scores.gather(1, s.data)
            case "r":
                scores = self.model(s, None, o, t).data
                score_of_expected = scores.gather(1, r.data)
            case "t":
                scores = self.model(s, r, None, t).data
                score_of_expected = scores.gather(1, o.data)
            case "Tf":
                t_s = t[:, :, 0]
                
                score_from_model = self.model(s, r, o, None)
                score_of_expected = torch.zeros(score_from_model.shape[0])
                for index in range(score_from_model.shape[0]):
                    score_of_expected[index] = score_from_model[index, t_s[index]]
                score_of_expected = score_of_expected.unsqueeze(-1)
                scores = score_from_model.view((score_from_model.shape[0], -1))
            case "Tt":
                t_e = t[:, :, 2]
                
                score_from_model = self.model(s, r, o, None)
                score_of_expected = torch.zeros(score_from_model.shape[0])
                for index in range(score_from_model.shape[0]):
                    score_of_expected[index] = score_from_model[index, t_e[index]]
                score_of_expected = score_of_expected.unsqueeze(-1)
                scores = score_from_model.view((score_from_model.shape[0], -1))

        return scores, score_of_expected

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
        
        simulated_fact = self.simulate_fact(head, relation, tail, time_from, time_to, target, answer)
        facts = self.fact_as_ids(simulated_fact)
        s, r, o, t = self.shred_fact(facts)
        scores, score_of_expected = self.get_scores(target, s, r, o, t)
        
        rank = self.get_rank(torch.reshape(scores, (-1,)), torch.reshape(score_of_expected, (-1,)).item())

        return int(rank)
