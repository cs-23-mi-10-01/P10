import torch
import numpy as np
import datetime
import os
from scripts import remove_unwanted_symbols_from_str, read_json, write_json
from dataset_handler.dataset_handler import DatasetHandler

class RankCalculator:
    def __init__(self, params, model, dataset_name, split = "original"):
        self.params = params
        self.model = model
        self.dataset = dataset_name
        self.split = split

        self.dataset_resource_folder = os.path.join(self.params.base_directory, "rank", "TimePlex", "resources", dataset_name, "split_" + split)

        self.dataset_handler = DatasetHandler(self.params, self.dataset)
        self.entity_map = read_json(os.path.join(self.dataset_resource_folder, "entity_map.json"))
        self.reverse_entity_map = self._create_reverse_map(self.entity_map)
        self.relation_map = read_json(os.path.join(self.dataset_resource_folder, "relation_map.json"))
        self.timestamp2id = read_json(os.path.join(self.dataset_resource_folder, "timestamp2id.json"))
        self.id2timestamp = self._create_reverse_map(self.timestamp2id)
        self.interval2id = read_json(os.path.join(self.dataset_resource_folder, "timestamp_interval2interval_id.json"))
        self.id2interval = self._create_reverse_map(self.interval2id)
        self.time_str2id = read_json(os.path.join(self.dataset_resource_folder, "time_str2id.json"))
        self.id2time_str = self._create_reverse_map(self.time_str2id)

    def _create_reverse_map(self, map):
        reverse_map = {}
        for key in map.keys():
            reverse_map[map[key]] = key
        return reverse_map

    def _get_rank(self, scores, score_of_expected):
        sum = 0

        for score in scores:
            if score > score_of_expected:
                sum += 1
        
        return sum + 1

    def _get_ent_id(self, entity):
        entity_id = self.dataset_handler.entity2id(entity)

        if entity_id in self.entity_map.keys():
            return self.entity_map[entity_id]
        else:
            return self.entity_map["<OOV>"]
        
    def _get_ent_from_id(self, id):
        dataset_id = self.reverse_entity_map[id]
        if dataset_id == "<OOV>":
            return "-"
        return self.dataset_handler.id2entity(dataset_id)

    def _get_rel_id(self, relation):
        return self.relation_map[self.dataset_handler.relation2id(relation)]
    
    def _get_rel_from_id(self, id):
        for key in self.relation_map.keys():
            if self.relation_map[key] == id:
                return self.dataset_handler.id2relation(key)
        
        return None
    
    def _get_time_id(self, year, month, day):
        if self.dataset in ['icews14']:
            interval_id = self._time_str_id(year, month, day, year, month, day)
            timestamp_interval = self.id2interval[interval_id]
            timestamp = timestamp_interval.split("(")[1].split(",")[0]
            id = self.timestamp2id[timestamp]
            return id
        elif self.dataset in ['wikidata12k', 'yago11k']:
            if str(year) in self.timestamp2id.keys():
                return self.timestamp2id[str(year)]
            else:
                return self.timestamp2id["UNK-TIME"]
            
    def _get_time_from_id(self, id):
        if self.dataset in ["icews14"]:
            timestamp = self.id2timestamp[id]
            if timestamp == "UNK-TIME":
                return "-"

            if timestamp in ["0", "3000"]:
                interval_id = self.interval2id[f"(0, 3000)"]
            else:
                interval_id = self.interval2id[f"({timestamp}, {timestamp})"]

            time_str = self.id2time_str[interval_id].split('\t')[0]
            return time_str
        
        if self.dataset in ["wikidata12k", "yago11k"]:
            timestamp = self.id2timestamp[id]
            if timestamp == "UNK-TIME":
                return "-"
            return timestamp
        
    def _interval_id(self, from_year, from_month, from_day, to_year, to_month, to_day):
        if self.dataset in ["icews14"]:
            from_timestamp = self.id2timestamp[self._get_time_id(from_year, from_month, from_day)]
            to_timestamp = self.id2timestamp[self._get_time_id(from_year, from_month, from_day)]

        if self.dataset in ["wikidata12k", "yago11k"]:
            from_timestamp = from_year
            to_timestamp = to_year
            
        if f"({from_timestamp}, {to_timestamp})" in self.interval2id.keys():
            return self.interval2id[f"({from_timestamp}, {to_timestamp})"]
        else:
            return self.interval2id["(UNK-TIME, UNK-TIME)"]
    
    def _year_to_iso_format(self, year):
        modified_year = str(year)
        if modified_year == '-':
            modified_year = "####"
        return self._to_iso_format(modified_year, "##", "##")
    
    def _to_iso_format(self, year, month, day):
        return f"{year}-{month}-{day}"
    
    def _time_str_id(self, from_year, from_month, from_day, to_year, to_month, to_day):
        if self.dataset in ["wikidata12k", "yago11k"]:
            from_iso = self._year_to_iso_format(from_year)
            to_iso = self._year_to_iso_format(to_year)
        elif self.dataset in ["icews14"]:
            from_iso = self._to_iso_format(from_year, from_month, from_day)
            to_iso = self._to_iso_format(from_year, from_month, from_day)
        
        if f"{from_iso}\t{to_iso}" in self.time_str2id.keys():
            return self.time_str2id[f"{from_iso}\t{to_iso}"]
        else:
            return self.time_str2id["####-##-##\t####-##-##"]
    
    def _all_entities(self):
        return [self.dataset_handler.id2entity(e) for e in self.entity_map.keys() if e != "<OOV>"]
    
    def _all_entity_ids(self):
        return self.entity_map.values()
    
    def _all_relations(self):
        return [self.dataset_handler.id2relation(r) for r in self.relation_map.keys()]
    
    def _all_relation_ids(self):
        return self.relation_map.values()
    
    def _all_timestamp_ids(self):
        return self.timestamp2id.values()
    
    def _number_of_timestamps(self):
        return len(self.timestamp2id) - 1
    
    def _if_year(self, year):
        if year == "UNK-TIME":
            return 0
        return year

    def _split_timestamp(self, element):
        if element == '-':
            return "UNK-TIME", "UNK-TIME", "UNK-TIME"
        if self.dataset in ['wikidata12k', 'yago11k']:
            return int(element), 0, 0
        else:
            dt = datetime.date.fromisoformat(element)
            return dt.year, dt.month, dt.day

    def _simulate_fact(self, head, relation, tail, time_from, time_to, target, answer):
        if time_from != "0":
            year_from, month_from, day_from = self._split_timestamp(time_from)
        if time_to != "0":
            year_to, month_to, day_to = self._split_timestamp(time_to)

        match target:
            case "h":
                sim_fact = [answer, relation, tail, year_from, month_from, day_from, year_to, month_to, day_to]
            case "r":
                sim_fact = [head, answer, tail, year_from, month_from, day_from, year_to, month_to, day_to]
            case "t":
                sim_fact = [head, relation, answer, year_from, month_from, day_from, year_to, month_to, day_to]
            case "Tf":
                ans_year, ans_month, ans_day = self._split_timestamp(answer)
                sim_fact = [head, relation, tail, ans_year, ans_month, ans_day, year_to, month_to, day_to]
            case "Tt":
                ans_year, ans_month, ans_day = self._split_timestamp(answer)
                sim_fact = [head, relation, tail, year_from, month_from, day_from, ans_year, ans_month, ans_day]
            case _:
                raise Exception("Unknown target")

        return sim_fact
    
    def _fact_as_ids(self, fact):
        return [self._get_ent_id(fact[0]),
                self._get_rel_id(fact[1]), 
                self._get_ent_id(fact[2]), 
                self._get_time_id(fact[3], fact[4], fact[5]), 
                self._if_year(fact[3]), 
                self._get_time_id(fact[6], fact[7], fact[8]), 
                self._if_year(fact[6]), 
                self._time_str_id(fact[3], fact[4], fact[5], fact[6], fact[7], fact[8]), 
                self._interval_id(fact[3], fact[4], fact[5], fact[6], fact[7], fact[8])]

    def _shred_fact(self, fact):

        s = np.array([fact[0]], dtype='int64')
        r = np.array([fact[1]], dtype='int64')
        o = np.array([fact[2]], dtype='int64')
        t = np.array([fact[3:]], dtype='int64')
        
        s = torch.autograd.Variable(torch.from_numpy(s).unsqueeze(1), requires_grad=False)
        r = torch.autograd.Variable(torch.from_numpy(r).unsqueeze(1), requires_grad=False)
        o = torch.autograd.Variable(torch.from_numpy(o).unsqueeze(1), requires_grad=False)
        t = torch.autograd.Variable(torch.from_numpy(t).unsqueeze(1), requires_grad=False)
        
        return s, r, o, t
    
    def _assign_ids(self, scores, target = "?"):
        ids = []
        match(target):
            case "h" | "t":
                ids = self._all_entity_ids()
            case "r":
                ids = self._all_relation_ids()
            case "Tf" | "Tt":
                ids = self._all_timestamp_ids()
        
        id_scores = []

        for id in ids:
            id_as_tensor = torch.tensor([[id]], dtype=torch.int64)
            score_as_float = scores.gather(1, id_as_tensor).item()
            id_scores.append((id, score_as_float))

        return id_scores
            

    def _get_scores(self, target, s, r, o, t):
        match target:
            case "h":
                score_tensor = self.model(None, r, o, t).data
            case "r":
                score_tensor = self.model(s, None, o, t).data
            case "t":
                score_tensor = self.model(s, r, None, t).data
            case "Tf":
                t_s = t[:, :, 0]
                
                score_from_model = self.model(s, r, o, None)
                score_of_expected = torch.zeros(score_from_model.shape[0])
                for index in range(score_from_model.shape[0]):
                    score_of_expected[index] = score_from_model[index, t_s[index]]
                score_of_expected = score_of_expected.unsqueeze(-1)
                score_tensor = score_from_model.view((score_from_model.shape[0], -1))
            case "Tt":
                t_e = t[:, :, 2]
                
                score_from_model = self.model(s, r, o, None)
                score_of_expected = torch.zeros(score_from_model.shape[0])
                for index in range(score_from_model.shape[0]):
                    score_of_expected[index] = score_from_model[index, t_e[index]]
                score_of_expected = score_of_expected.unsqueeze(-1)
                score_tensor = score_from_model.view((score_from_model.shape[0], -1))
               
        id_scores = self._assign_ids(score_tensor, target)

        return id_scores
    
    def _construct_facts_scores(self, head, relation, tail, time_from, time_to, id_scores, target):
        fact_scores = {}

        for id, score in id_scores:
            match target:
                case "h":
                    fact_scores[(self._get_ent_from_id(id), relation, tail, time_from, time_to)] = score
                case "r":
                    fact_scores[(head, self._get_rel_from_id(id), tail, time_from, time_to)] = score
                case "t":
                    fact_scores[(head, relation, self._get_ent_from_id(id), time_from, time_to)] = score
                case "Tf":
                    fact_scores[(head, relation, tail, self._get_time_from_id(id), time_to)] = score
                case "Tt":
                    fact_scores[(head, relation, tail, time_from, self._get_time_from_id(id))] = score

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
        elif time_to == "0":
            target = "Tt"

        time_to_not_blank = time_to
        if self.dataset in ["icews14"]:
            if target == "Tf":
                time_to_not_blank = answer
            else:
                time_to_not_blank = time_from
        
        simulated_fact = self._simulate_fact(head, relation, tail, time_from, time_to_not_blank, target, answer)
        fact = self._fact_as_ids(simulated_fact)
        s, r, o, t = self._shred_fact(fact)
        id_scores = self._get_scores(target, s, r, o, t)
        fact_scores = self._construct_facts_scores(head, relation, tail, time_from, time_to, id_scores, target)

        return fact_scores

    def rank_of_correct_prediction(self, fact_scores, correct_fact):
        if correct_fact not in fact_scores.keys():
            return 10000

        return self._get_rank(fact_scores.values(), fact_scores[correct_fact])
    
    def best_prediction(self, fact_scores):
        highest_scoring_fact_key = max(fact_scores, key = lambda key: fact_scores[key])
        return highest_scoring_fact_key[3]
