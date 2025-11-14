import uuid

import jsonlines
import numpy as np
from llamea import Solution


class LogDeserializer(object):

    def __init__(self, logfile: str):
        self.logfile = logfile

    def read_solution(self, data: dict) -> Solution:
        sol = Solution(
            code=data.get("code", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            configspace=None, # Not implemented
            generation=data.get("generation", 0),
            parent_ids=data.get("parent_ids", []),
            operator=data.get("operator"),
            task_prompt=data.get("task_prompt", ""),
        )

        sol.id = data.get("id", str(uuid.uuid4()))
        sol.fitness = data.get("fitness", -np.inf)
        sol.feedback = data.get("feedback", "")
        sol.error = data.get("error", "")
        sol.metadata = data.get("metadata", {})
        return sol

    def read(self) -> list[Solution]:
        self.solutions = []
        with jsonlines.open(self.logfile) as reader:
            for obj in reader:
                self.solutions.append(self.read_solution(obj))
        return self.solutions
