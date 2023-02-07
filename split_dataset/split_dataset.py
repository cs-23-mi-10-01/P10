
import os
import csv

from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import TemporalData
from torch import Tensor

class SplitDataset:
    def __init__(self, params):
        self.params = params
        self.base_directory = params.base_directory
        self.dataset = params.dataset
        
        self.entities = []
        self.relations = []
        self.timestamps = []

    def split(self):
        dataset_path = os.path.join(self.base_directory, "datasets", self.dataset, "full.txt")
        rows = []
        with open(dataset_path, encoding='utf-8') as full_dataset:
            records = csv.DictReader(full_dataset, delimiter='\t')
            for row in records:
                rows.append(row)
        
        data = TemporalData(
            src=self._to_tensor([r['head'] for r in rows], self.entities),
            msg=self._to_tensor([r['relation'] for r in rows], self.relations),
            dst=self._to_tensor([r['tail'] for r in rows], self.entities),
            t=self._to_tensor([r['timestamp'] for r in rows], self.timestamps)
        )

        rls = RandomLinkSplit(is_undirected=False,
                            add_negative_train_samples=False,
                            num_val=0.15,
                            num_test=0.15,
                            )
        train_data, val_data, test_data = rls(data)
        print("test")

    def _to_tensor(self, column, element_arr):
        tensor_arr = []
        for element in column:
            tensor_arr.append(self._get_id(element, element_arr))
        return Tensor(tensor_arr)

    def _get_id(self, element, element_arr):
        if element in element_arr:
            return element_arr.index(element)
        else:
            element_arr.append(element)
            return len(element_arr)
