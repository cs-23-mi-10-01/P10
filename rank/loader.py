
import torch
import sys
from scripts import remove_unwanted_symbols
from rank.de_simple import de_transe, de_simple, de_distmult, dataset, params
from rank.TERO import TERO_model, Dataset
from rank.TFLEX.tflex import FLEX
from rank.TimePlex import models as Timeplex_model, kb

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
        elif self.embedding in ["TFLEX"]:
            pass
        elif self.embedding in ["TimePlex"]:
            sys.modules['model']=Timeplex_model
            sys.modules['kb']=kb

        if self.embedding in ["DE_TransE", "DE_SimplE", "DE_DistMult", "TERO", "ATISE"]:
            model = torch.load(self.model_path, map_location="cpu")
        elif self.embedding in ["TFLEX"]:
            state_dict = torch.load(self.model_path, map_location="cpu")
            
            model = FLEX()
            model.load_state_dict(self, state_dict["model_state_dict"])

        elif self.embedding in ["TimePlex"]:
            map_location = None if False else 'cpu'
            model = torch.load(self.model_path, map_location = map_location)
            # state_dict = torch.load(self.model_path)
            # model = Timeplex_model.TimePlex(**state_dict['model_arguments'])
            # model.load_state_dict(state_dict['model_weights'])
            #state_dict = test.load_state(self.model_path)
            #state_dict = torch.load(self.model_path, map_location="cpu")
            


        sys.modules = old_modules

        if self.embedding in ["DE_TransE", "DE_SimplE", "DE_DistMult"]:
            remove_unwanted_symbols(model.module.dataset.ent2id)
            remove_unwanted_symbols(model.module.dataset.rel2id)
        elif self.embedding in ['TERO', 'ATISE']:
            remove_unwanted_symbols(model.kg.entity_dict)
            remove_unwanted_symbols(model.kg.relation_dict)
            model.gpu = False

        return model
