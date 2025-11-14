import csv

from llamea import ExperimentLogger


class ArenaExperimentLogger(ExperimentLogger):

    def __init__(self, name: str):
        super().__init__(name, )

    def log_match(self, dataset_name: str, original_id: str, new_id: str, fitness_delta: float):
        file_path = f"{self.dirname}/matchlog.csv"
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([dataset_name, original_id, new_id, fitness_delta])

    def log_conversation(self, role, content, cost=0.0, tokens=0): # TODO: maybe remove
        return super().log_conversation(role, content)