from llamea import Solution


class MatchResult(object):

    def __init__(self, dataset: str, parent_ids: list[str], parent_fitness: float, new_solution: Solution, fitness_delta: float):
        self.dataset = dataset
        self.parent_ids = parent_ids
        self.parent_fitness = parent_fitness
        self.new_solution = new_solution
        self.fitness_delta = fitness_delta

    def to_dict(self):
        return {
            "dataset": self.dataset,
            "parent_ids": self.parent_ids,
            "parent_fitness": self.parent_fitness,
            "new_solution": self.new_solution.to_dict(),
            "fitness_delta": self.fitness_delta
        }