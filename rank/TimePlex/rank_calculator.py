import torch
import numpy as np
import time
import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
from scripts import year_to_iso_format
from dataset_handler.dataset_handler import DatasetHandler

class RankCalculator:
    def __init__(self, params, model, dataset_name):
        self.params = params
        self.model = model
        self.dataset_name = dataset_name

        if self.dataset_name in ['icews14']:
            self.start_sim_date = datetime.date(2014, 1, 1)
            self.end_sim_date = datetime.date(2015, 1, 1)
            self.delta_sim_date = relativedelta(days=1)
        elif self.dataset_name in ['wikidata12k']:
            self.start_sim_date = datetime.date(1, 1, 1)
            self.end_sim_date = datetime.date(2021, 1, 1)
            self.delta_sim_date = relativedelta(years=1)
        elif self.dataset_name in ['yago11k']:
            self.start_sim_date = datetime.date(1, 1, 1)
            self.end_sim_date = datetime.date(2845, 1, 1)
            self.delta_sim_date = relativedelta(years=1)

        self.dataset_handler = DatasetHandler(self.params, self.dataset_name)

    def get_rank(self, scores, score_of_expected):
        return torch.sum((scores > score_of_expected).float()).item() + 1

    def get_ent_id(self, entity):
        return int(self.dataset_handler.entity2id(entity))

    def get_rel_id(self, relation):
        return int(self.dataset_handler.relation2id(relation))
    
    def split_timestamp(self, element):
        if self.dataset_name in ['wikidata12k', 'yago11k']:
            modified_date = year_to_iso_format(element)
        else:
            modified_date = element

        dt = datetime.date.fromisoformat(modified_date)
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
                sim_facts = [[i, relation, tail, year_from, month_from, day_from, year_to, month_to, day_to] for i in range(self.kg.n_entity)]
                sim_facts = [[self.get_ent_id(answer), relation, tail, year_from, month_from, day_from, year_to, month_to, day_to]] + sim_facts
            case "r":
                sim_facts = [[head, i, tail, year_from, month_from, day_from, year_to, month_to, day_to] for i in range(self.kg.n_relation)]
                sim_facts = [[head, self.get_rel_id(answer), tail, year_from, month_from, day_from, year_to, month_to, day_to]] + sim_facts
            case "t":
                sim_facts = [[head, relation, i, year_from, month_from, day_from, year_to, month_to, day_to] for i in range(self.kg.n_entity)]
                sim_facts = [[head, relation, self.get_ent_id(answer), year_from, month_from, day_from, year_to, month_to, day_to]] + sim_facts
            case "Tf":
                sim_facts = []
                sim_date = self.start_sim_date
                while sim_date != self.end_sim_date:
                    sim_year = sim_date.year
                    sim_month = sim_date.month
                    sim_day = sim_date.day
                    sim_facts.append((head, relation, tail, sim_year, sim_month, sim_day, year_to, month_to, day_to))
                    sim_date = sim_date + self.delta_sim_date

                sim_year, sim_month, sim_day = self.split_timestamp(answer)
                sim_facts = [[head, relation, tail, sim_year, sim_month, sim_day, year_to, month_to, day_to]] + sim_facts
            case _:
                raise Exception("Unknown target")

        return sim_facts

    def shred_fact(self, facts):
        s = np.array([facts[0]])
        r = np.array([facts[1]])
        o = np.array([facts[2]])
        t = np.array([facts[3:]])
        
        s = torch.autograd.Variable(torch.from_numpy(s).unsqueeze(1), requires_grad=False)
        r = torch.autograd.Variable(torch.from_numpy(r).unsqueeze(1), requires_grad=False)
        o = torch.autograd.Variable(torch.from_numpy(o).unsqueeze(1), requires_grad=False)
        t = torch.autograd.Variable(torch.from_numpy(t).unsqueeze(1), requires_grad=False)
        
        return s, r, o, t

    def get_rank_of(self, head, relation, tail, time_from, time_to, answer):
        fact = []
        if head == "0":
            fact.append(self.get_ent_id(answer))
        else:
            fact.append(self.get_ent_id(head))

        if relation == "0":
            fact.append(self.get_rel_id(answer))
        else:
            fact.append(self.get_rel_id(relation))

        if tail == "0":
            fact.append(self.get_ent_id(answer))
        else:
            fact.append(self.get_ent_id(tail))

        if time_from == "0":
            fact.extend(self.split_timestamp(answer))
        else:
            fact.extend(self.split_timestamp(time_from))
            
        if time_to == "0":
            fact.extend(self.split_timestamp(answer))
        else:
            fact.extend(self.split_timestamp(time_to))
        
        s, r, o, t = self.shred_fact(fact)
        scores, score_of_expected = self.model.forward(s, r, o, t,
                                                       flag_s_o = 1 if tail == "0" else 0,
                                                       flag_r = 1 if relation == "0" else 0, 
                                                       flag_t = 1 if time_from == "0" else 0, 
                                                       flag_tp = 0)
        rank = self.get_rank(scores, score_of_expected)

        return int(rank)
