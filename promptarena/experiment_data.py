import random
from collections import defaultdict
from typing import Tuple

from llamea import Solution

from misc.log_deserializer import LogDeserializer


class ExperimentData(object):

    def __init__(self, name: str, solutions: list[Solution]):
        self.name = name
        self.solutions = solutions
        self.generation_map = self._group_by_generation()
        self.generations = len(self.generation_map)
    
    @staticmethod
    def load_log(logfile: str) -> "ExperimentData":
        deserializer = LogDeserializer(logfile)
        solutions = deserializer.read()
        return ExperimentData(logfile, solutions)
    
    def sample_non_final_generation(self) -> list[Solution]:
        if not self.generation_map or self.generations <= 1:
            return []
        
        non_final_gen_keys = [g for g in self.generation_map.keys() if g != max(self.generation_map.keys())]
        return self.generation_map[random.choice(non_final_gen_keys)]

    def _group_by_generation(self):
        gen_map = defaultdict(list)
        for sol in self.solutions:
            gen_map[sol.generation].append(sol)
        return dict(gen_map)
    


class ExperimentDataSet(object):
    def __init__(self, experiments: list[ExperimentData]):
        self.experiments = experiments

    @staticmethod
    def load_logs(logfiles: list[str]) -> "ExperimentDataSet":
        experiments = [ExperimentData.load_log(file) for file in logfiles]
        return ExperimentDataSet(experiments)

    def sample_experiment(self) -> ExperimentData:
        return random.choice(self.experiments)

    def sample_non_final_generation(self) -> Tuple[str, list[Solution]]:
        data = random.choice(self.experiments)
        generation = data.sample_non_final_generation()
        return data.name, generation