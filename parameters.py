import os
from timer import Timer


class Parameters:
    def __init__(self, args):
        self.task = args.task
        self.datasets = [args.dataset]
        self.embeddings = [args.embedding]
        self.splits = [args.split]

        self.base_directory = os.path.abspath(os.path.dirname(__file__))
        self.add_to_result = args.add_to_result
        self.timer = Timer()
