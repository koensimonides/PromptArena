import contextlib
from typing import Tuple

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
        self.prompt_constructor: PromptConstructor = constructor
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
        
    def logevent(self, event):
        print(event)

    def evaluate_fitness(self, individual: Solution) -> Solution:
        with contextlib.redirect_stdout(None):
            updated_individual = self.f(individual) # None parameter is logger in LLaMEA

        return updated_individual
    
    def run_match(self, data_name: str, parent_population: list[Solution], original: Solution) -> MatchResult:
        """
        Run a single match by generating and evaluating a new solution.

        Args:
            data_name (str): Identifier of the dataset.
            parent_population (list[Solution]): Candidate parents for selection.
            original (Solution): Baseline solution used for comparison.

        Returns:
            MatchResult: Match data and new solution
        """
        
        parents = self._select_parents(parent_population, original)

        if len(parents) != self.prompt_constructor.parent_count:
            self.logevent(f"Failed to find parents for original {original.id}")
            return None

        parent_fitness = max([parent.fitness for parent in parents]) # Use best parent fitness
        prompt_text = self.prompt_constructor([parent.copy() for parent in parents], parent_population)

        session_messages = [
            {"role": "user", "content": prompt_text},
        ]

        new_solution = parents[0].empty_copy()
        try:
            new_solution = self.llm.sample_solution(
                session_messages=session_messages,
                parent_ids=[subj.id for subj in parents]
            )

            # TODO fix: this is caused by a np version mismatch, don't know how to fix yet
            new_solution.code = new_solution.code.replace("np.Inf", "np.inf")
            # ----------

            new_solution = self.evaluate_fitness(new_solution)
        except Exception as e:
            new_solution.set_scores(
                -np.inf, f"An exception occurred: {e.__repr__()}.", e
            )

        new_solution.generation = original.generation
        fitness_delta = self._compare(parent_fitness, new_solution)

        return MatchResult(data_name, original.id, parent_fitness, new_solution, fitness_delta)
 
    def run(self) -> list[MatchResult]:
        """
        Execute all scheduled matches for this experiment.

        Returns:
            list[MatchResult]: Collected results for each match.
        """
            
        self.logevent(
            f"{self.experiment_name} | running {self.budget} matches"
        )

        self.results = []

        for i in range(self.budget):
            # Sample match
            experiment = self.data.sample_experiment()
            parent_population = experiment.sample_non_final_generation()
            child_generation = parent_population[0].generation + 1
            original = np.random.choice(experiment.generation_map[child_generation])

            # Run match
            result: MatchResult | None = self.run_match(experiment.name, parent_population, original)
            if result == None: # TODO, should in theory never happen
                continue

            # Log results
            if self.log:
                self.logger.log_population([result.new_solution])
                self.logger.log_match(result)

            self.results.append(result)

            error_summary = f", Error: {trim_lines(result.new_solution.error, 100)}" if result.new_solution.error != "" else ""
            self.logevent(
                f"Match {i} fitness delta: {result.fitness_delta}{error_summary}"
            )
        
        return self.results

    def _select_parents(self, original_parent_population: list[Solution], original: Solution) -> list[Solution]:
        """
        Select a parent set matching the required parent count.

        Args:
            original_parent_population (list[Solution]): Full available population.
            original (Solution): Original solution containing its parent IDs.

        Returns:
            list[Solution]: Selected parent solutions sized to `parent_count`.
        """
            
        target_count = self.prompt_constructor.parent_count

        if (target_count > len(original_parent_population)):
            raise ValueError(f"Cannot satisfy prompt subjects, required {target_count} but population is of size {len(original_parent_population)}")

        original_parents = [parent for parent in original_parent_population if parent.id in original.parent_ids]

        new_parents: list[Solution] = []
        if (len(original_parents) > target_count):
            # Original prompt used more parents than current prompt, pick required count from these
            new_parents = np.random.choice(original_parents, size=target_count, replace=False)
        elif (len(original_parents) < target_count):
            # Original prompt used less parents than current prompt, add extra from remaining population
            remaining_parent_pop = [parent for parent in original_parent_population if parent.id not in original.parent_ids]
            remaining_count = target_count - len(original_parents)
            extra = np.random.choice(remaining_parent_pop, size=remaining_count, replace=False).tolist()
            new_parents = original_parents + extra
        else:
            # Original prompt used the same amount of parents as current prompt, copy that selection
            new_parents = original_parents

        return new_parents
    
    def _compare(self, parent_fitness: float, new: Solution) -> float:
        # Determine fitness delta
        # both >= 0         -> new fitness - old fitness
        # new >= 0, old bad -> new fitness 
        # new bad, old > 0  -> -old fitness
        # both bad          -> 0
        fitness_delta = 0.0
        if new.fitness >= 0 and parent_fitness >= 0:
            fitness_delta = new.fitness - parent_fitness
        elif new.fitness >= 0:
            fitness_delta = new.fitness
        elif parent_fitness > 0:
            fitness_delta = -parent_fitness

        return fitness_delta