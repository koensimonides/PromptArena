import random

from llamea import Solution

from promptarena.prompt_constructor import PromptConstructor


# Copied from LLaMEA crossover prompt constructor function
# https://github.com/XAI-liacs/LLaMEA/blob/e4eafe7d684a48839fef65fb7bc64be42a57e8cd/llamea/llamea.py#L423
class CrossoverPrompt(PromptConstructor):

    def __init__(
            self,
            task_prompt = "",
            role_prompt = "",
            crossover_prompts = None,
            parent_count: int = 2,
            include_population_summary: bool = True
            ):
        super().__init__(parent_count)

        if task_prompt == "":
            self.task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code to minimize the function value. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.

Give an excellent and novel heuristic algorithm to solve this task.
"""
        else:
            self.task_prompt = task_prompt

        self.role_prompt = role_prompt
        if role_prompt == "":
            self.role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."

        self.crossover_prompts = crossover_prompts
        if crossover_prompts == None:
            self.crossover_prompts = [
                "Combine the algorithmic structures of the following two optimization methods by simulating a natural genetic crossover, producing a hybrid metaheuristic that inherits key traits from both parents.",
            ]

        self.include_population_summary = include_population_summary

    def make_prompt(self, parents: list[Solution], population: list[Solution]) -> str:
        # Generate the current population summary
        population_summary = "\n".join([ind.get_summary() for ind in population])
        solutions = [p.code for p in parents]
        descriptions = [p.description for p in parents]
        feedbacks = [p.feedback for p in parents]
        task_prompt = random.choice(self.crossover_prompts)
        
        summary_prompt = f"""
The current population of algorithms already evaluated (name, description, score) is:
{population_summary}

""" if self.include_population_summary else ""
        
        final_prompt = f"""{task_prompt}
{summary_prompt}
The selected solutions to apply crossover are:
"""
        for i in range(len(solutions)):
            final_prompt += f"""
{descriptions[i]}

With code:
{solutions[i]}

{feedbacks[i]}

"""

        return self.role_prompt + "\n\n" + final_prompt