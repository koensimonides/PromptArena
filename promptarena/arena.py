import contextlib

import numpy as np
from llamea import LLM, Solution

from promptarena.experiment_data import ExperimentDataSet
from promptarena.match_result import MatchResult
from promptarena.prompt_constructor import PromptConstructor
from promptarena.logger import ArenaExperimentLogger
from promptarena.utils import make_filename_safe, trim_lines


class PromptArena(object):

    def __init__(
            self,
            f,
            constructor: PromptConstructor,
            llm: LLM,
            data: ExperimentDataSet,
            experiment_name: str = "",
            budget: int = 100,
            log: bool = True
            ):
        self.f = f
        self.constructor: PromptConstructor = constructor
        self.llm: LLM = llm
        self.data: ExperimentDataSet = data
        self.experiment_name: str = experiment_name
        self.budget: int = budget
        self.log: bool = log

        self.model_name = self.llm.model.replace(":", "_")
        self.log_name = make_filename_safe(f"Arena-{self.experiment_name}-{self.model_name}")
        self.results: list[MatchResult] = []

        if self.log:
            self.logger = ArenaExperimentLogger(self.log_name.lower())
            self.llm.set_logger(self.logger)
        else:
            self.logger = None
    
    def evaluate_fitness(self, individual: Solution) -> Solution:
        with contextlib.redirect_stdout(None):
            updated_individual = self.f(individual) # None parameter is logger in LLaMEA

        return updated_individual
    
    def run_match(self, data_name:str, parent_population: list[Solution], child: Solution) -> MatchResult:
        subjects = self._select_subjects(parent_population, child)

        # Make copy of subjects because llamea does so, not sure why TODO?
        prompt_text = self.constructor([subj.copy() for subj in subjects], parent_population)

        session_messages = [
            {"role": "user", "content": prompt_text},
        ]

        new_child = subjects[0].empty_copy()
        try:
            new_child = self.llm.sample_solution(
                session_messages=session_messages,
                parent_ids=[subj.id for subj in subjects]
            )

            # TODO fix: this is caused by a np version mismatch, don't know how to fix yet
            new_child.code = new_child.code.replace("np.Inf", "np.inf")
            # ----------

            new_child = self.evaluate_fitness(new_child)
        except Exception as e:
            new_child.set_scores(
                -np.inf, f"An exception occurred: {e.__repr__()}.", e
            )

        new_child.generation = child.generation

        fitness_delta = self._compare_fitness(new_child.fitness, child.fitness)
        result = MatchResult(data_name, child.id, new_child, fitness_delta)
        return result
    
    def logevent(self, event):
        print(event)
    
    def run(self) -> list[MatchResult]:
        self.logevent(
            f"{self.experiment_name} | running {self.budget} matches"
        )

        self.results = []

        for i in range(self.budget):
            experiment = self.data.sample_experiment()
            parent_population = experiment.sample_non_final_generation()
            child_generation = parent_population[0].generation + 1
            child = np.random.choice(experiment.generation_map[child_generation])

            result: MatchResult = self.run_match(experiment.name, parent_population, child)

            if self.log:
                self.logger.log_population([result.new_solution])
                self.logger.log_match(result.original_data, result.original_id, result.new_solution.id, result.fitness_delta)

            self.results.append(result)

            error_summary = f", Error: {trim_lines(result.new_solution.error, 100)}" if result.new_solution.error != "" else ""
            self.logevent(
                f"Match {i} fitness delta: {result.fitness_delta}{error_summary}"
            )
        
        return self.results

    def _select_subjects(self, parent_population: list[Solution], child: Solution) -> list[Solution]:
        target_count = self.constructor.subject_count

        if (target_count > len(parent_population)):
            raise ValueError(f"Cannot satisfy prompt subjects, required {target_count} but population is of size {len(parent_population)}")

        parents = [parent for parent in parent_population if parent.id in child.parent_ids]

        subjects: list[Solution] = []
        if (len(parents) > target_count):
            # Original prompt used more parents than current prompt, pick required count from these
            subjects = np.random.choice(parents, size=target_count, replace=False)
        elif (len(parents) < target_count):
            # Original prompt used less parents than current prompt, add extra from remaining population
            remaining_parent_pop = [parent for parent in parent_population if parent.id not in child.parent_ids]
            remaining_count = target_count - len(parents)
            subjects = parents + np.random.choice(remaining_parent_pop, size=remaining_count, replace=False) 
        else:
            # Original prompt used the same amount of parents as current prompt, copy that selection
            subjects = parents

        return subjects
    
    def _compare_fitness(self, new_fitness: float, old_fitness: float) -> float:
        # Determine fitness delta
        # both >= 0 -> new fitness - old fitness
        # new >= 0, old bad -> new fitness 
        # new bad, old > 0 -> -old fitness
        # both bad -> 0
        fitness_delta = 0.0
        if new_fitness >= 0:
            fitness_delta = new_fitness - old_fitness if old_fitness >= 0 else new_fitness
        elif old_fitness > 0: # if old was 0, it will default to 0.0 (-0.0 if we use >=)
            fitness_delta = -old_fitness
        
        return fitness_delta