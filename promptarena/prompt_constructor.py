from abc import ABC, abstractmethod

from llamea import Solution


class PromptConstructor(ABC):

    def __init__(self, subject_count: int):
        self.subject_count = subject_count

    def __call__(self, subjects: list[Solution], population: list[Solution]) -> str:
        if (len(subjects) != self.subject_count):
            raise ValueError(f"Invalid subject count, expected {self.subject_count} but recieved {len(subjects)}")
        
        prompt = self.make_prompt(subjects, population)
        return prompt

    @abstractmethod
    def make_prompt(self, subjects: list[Solution], population: list[Solution]) -> str:
        pass