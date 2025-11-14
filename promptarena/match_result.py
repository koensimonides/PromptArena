from llamea import Solution


class MatchResult(object):

    def __init__(self, original_data: str, original_id: str, new_solution: Solution, fitness_delta: float):
        self.original_data = original_data
        self.original_id = original_id
        self.new_solution = new_solution
        self.fitness_delta = fitness_delta

    def to_dict(self):
        return {
            "original_data": self.original_data,
            "original_id": self.original_id,
            "new_solution": self.new_solution.to_dict(),
            "fitness_delta": self.fitness_delta
        }