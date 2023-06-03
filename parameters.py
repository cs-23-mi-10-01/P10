import os
from timer import Timer


class Parameters:
    def __init__(self, args):
        self.task = args.task
        self.datasets = [args.dataset]
        self.embeddings = [args.embedding]
        self.splits = [args.split]
        self.verbose = args.summary

        self.base_directory = os.path.abspath(os.path.dirname(__file__))
        self.timer = Timer()
