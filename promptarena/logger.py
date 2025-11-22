import csv

from llamea import ExperimentLogger

from promptarena.match_result import MatchResult


class ArenaExperimentLogger(ExperimentLogger):

    def __init__(self, name: str):
        super().__init__(name, )

    def log_match(self, match_result: MatchResult):
        file_path = f"{self.dirname}/matchlog.csv"
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([match_result.original_data, match_result.original_id, match_result.parent_fitness, match_result.new_solution.id, match_result.fitness_delta])

    # To handle newer log_conversation function in iohblade
    # Can possibly be removed
    def log_conversation(self, role, content, cost=0.0, tokens=0):
        return super().log_conversation(role, content)