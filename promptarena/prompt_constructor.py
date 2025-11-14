from abc import ABC, abstractmethod

from llamea import Solution


class PromptConstructor(ABC):

    def __init__(self, parent_count: int):
        self.parent_count = parent_count

    def __call__(self, parents: list[Solution], population: list[Solution]) -> str:
        if (len(parents) != self.parent_count):
            raise ValueError(f"Invalid parent count, expected {self.parent_count} but recieved {len(parents)}")
        
        prompt = self.make_prompt(parents, population)
        return prompt

    @abstractmethod
    def make_prompt(self, parents: list[Solution], population: list[Solution]) -> str:
        pass