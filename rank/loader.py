
import torch
import sys
from scripts import remove_unwanted_symbols
from rank.de_simple import de_transe, de_simple, de_distmult, dataset, params
from rank.TERO import TERO_model, Dataset, Dataset_YG
from rank.TFLEX.tflex import FLEX
from rank.TimePlex import models as Timeplex_model

class Loader:
    def __init__(self, params, model_path, embedding):
        self.params = params
        self.model_path = model_path
        self.embedding = embedding

    def load(self):
        old_modules = sys.modules

        if self.embedding in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
            sys.modules['de_transe'] = de_transe
            sys.modules['de_simple'] = de_simple
            sys.modules['de_distmult'] = de_distmult
            sys.modules['dataset'] = dataset
            sys.modules['params'] = params
        elif self.embedding in ["TERO", "ATISE"]:
            sys.modules['model'] = TERO_model
            sys.modules['Dataset'] = Dataset
            sys.modules['Dataset_YG'] = Dataset_YG
        elif self.embedding in ["TFLEX"]:
            pass
        elif self.embedding in ["TimePlex"]:
            sys.modules['model']=Timeplex_model

        if self.embedding in ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE","TimePlex"]:
            model = torch.load(self.model_path, map_location="cpu")
        elif self.embedding in ["TFLEX"]:
            state_dict = torch.load(self.model_path, map_location="cpu")
            
            model = FLEX()
            model.load_state_dict(state_dict["model_state_dict"])

        sys.modules = old_modules

        if self.embedding in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
            remove_unwanted_symbols(model.module.dataset.ent2id)
            remove_unwanted_symbols(model.module.dataset.rel2id)
        elif self.embedding in ['TERO', 'ATISE']:
            remove_unwanted_symbols(model.kg.entity_dict)
            remove_unwanted_symbols(model.kg.relation_dict)
            model.gpu = False

        return model
