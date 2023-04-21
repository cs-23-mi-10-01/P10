import numpy as np
import torch
import datetime
from datetime import date
from scripts import remove_unwanted_symbols_from_str, year_to_iso_format
from dateutil.relativedelta import relativedelta


class RankCalculator:
    def __init__(self, params, model, dataset_name):
        self.params = params
        self.dataset = model.module.dataset
        self.model = model
        self.dataset_name = dataset_name

        self.num_of_ent = self.dataset.numEnt()
        self.num_of_rel = self.dataset.numRel()

        if self.dataset_name in ['icews14']:
            self.start_sim_date = date(2014, 1, 1)
            self.end_sim_date = date(2015, 1, 1)
            self.delta_sim_date = datetime.timedelta(days=1)
        elif self.dataset_name in ['wikidata12k']:
            self.start_sim_date = date(1, 1, 1)
            self.end_sim_date = date(2021, 1, 1)
            self.delta_sim_date = relativedelta(years=1)
        elif self.dataset_name in ['yago11k']:
            self.start_sim_date = date(1, 1, 1)
            self.end_sim_date = date(2845, 1, 1)
            self.delta_sim_date = relativedelta(years=1)


    def get_rank(self, sim_scores):  # assuming the test fact is the first one
        return (sim_scores > sim_scores[0]).sum() + 1

    def split_timestamp(self, element):
        if self.dataset_name in ['wikidata12k', 'yago11k']:
            modified_date = year_to_iso_format(element)
        else:
            modified_date = element

        dt = date.fromisoformat(modified_date)
        return dt.year, dt.month, dt.day

    def shred_facts(self, facts): #takes a batch of facts and shreds it into its columns
        device = torch.device("cpu")
        heads = torch.tensor(facts[:, 0]).long().to(device)
        rels = torch.tensor(facts[:, 1]).long().to(device)
        tails = torch.tensor(facts[:, 2]).long().to(device)
        years = torch.tensor(facts[:, 3]).float().to(device)
        months = torch.tensor(facts[:, 4]).float().to(device)
        days = torch.tensor(facts[:, 5]).float().to(device)
        return heads, rels, tails, years, months, days

    def get_ent_id(self, entity):
        entity_id = self.dataset.getEntID(remove_unwanted_symbols_from_str(entity))
        if entity_id >= self.num_of_ent:
            raise Exception("Fact contains an entity that is not seen in the training set (" + str(entity) + ")")
        return entity_id

    def get_rel_id(self, relation):
        rel_id = self.dataset.getRelID(remove_unwanted_symbols_from_str(relation))
        if rel_id >= self.num_of_rel:
            raise Exception("Fact contains a relation that is not seen in the training set (" + str(relation) + ")")
        return rel_id

    def simulate_facts(self, head, relation, tail, time, target, answer):
        if head != "0":
            head = self.get_ent_id(head)
        if relation != "0":
            relation = self.get_rel_id(relation)
        if tail != "0":
            tail = self.get_ent_id(tail)
        if time != "0":
            year, month, day = self.split_timestamp(time)

        match target:
            case "h":
                sim_facts = [(i, relation, tail, year, month, day) for i in range(self.num_of_ent)]
                sim_facts = [(self.get_ent_id(answer), relation, tail, year, month, day)] + sim_facts
            case "r":
                sim_facts = [(head, i, tail, year, month, day) for i in range(self.num_of_rel)]
                sim_facts = [(head, self.get_rel_id(answer), tail, year, month, day)] + sim_facts
            case "t":
                sim_facts = [(head, relation, i, year, month, day) for i in range(self.num_of_ent)]
                sim_facts = [(head, relation, self.get_ent_id(answer), year, month, day)] + sim_facts
            case "T":
                sim_facts = []
                sim_date = self.start_sim_date
                while sim_date != self.end_sim_date:
                    year = sim_date.year
                    month = sim_date.month
                    day = sim_date.day
                    sim_facts.append((head, relation, tail, year, month, day))
                    sim_date = sim_date + self.delta_sim_date

                year, month, day = self.split_timestamp(answer)
                sim_facts = [(head, relation, tail, year, month, day)] + sim_facts
            case _:
                raise Exception("Unknown target")

        return sim_facts

    def simulate_fact_scores(self, head, relation, tail, time_from, time_to, answer):
        target = "?"
        if head == "0":
            target = "h"
        elif relation == "0":
            target = "r"
        elif tail == "0":
            target = "t"
        elif time_from == "0":
            target = "T"

        facts = self.simulate_facts(head, relation, tail, time_from, target, answer)
        heads, rels, tails, years, months, days = self.shred_facts(np.array(facts))
        sim_scores = self.model.module(heads, rels, tails, years, months, days).cpu().data.numpy()
        
        scored_simulated_facts = []
        for fact, score in zip(facts, sim_scores):
            scored_simulated_facts.append([fact[0], fact[1], fact[2], fact[3], fact[4], fact[5], score])

        return scored_simulated_facts
    
    def rank_of_correct_prediction(self, fact_scores):
        return self.get_rank([fact[6] for fact in fact_scores])
    
    def best_prediction(self, fact_scores):
        scores = [fact[6] for fact in fact_scores]
        highest_score = max(scores)
        pred = fact_scores[scores.index(highest_score)][0:6]
        return date(pred[3], pred[4], pred[5]).isoformat()
