import random

from llamea import Solution

from promptarena.prompt_constructor import PromptConstructor


# Copied from LLaMEA prompt constructor function
# https://github.com/XAI-liacs/LLaMEA/blob/9209b25b1fd11744f78a19df5873aad23a4407cf/llamea/llamea.py#L453
class ClassicPrompt(PromptConstructor):

    def __init__(
            self,
            task_prompt = "",
            role_prompt = "",
            mutation_prompts = None
            ):
        super().__init__(1)

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

        self.mutation_prompts = mutation_prompts
        if mutation_prompts == None:
            self.mutation_prompts = [
                "Refine the strategy of the selected solution to improve it.",  # small mutation
                # "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
            ]

    def make_prompt(self, parents: list[Solution], population: list[Solution]) -> str:
        individual: Solution = parents[0]

        # Generate the current population summary
        population_summary = "\n".join([ind.get_summary() for ind in population])
        solution = individual.code
        description = individual.description
        feedback = individual.feedback
        error_message = ""
        if individual.error:
            error_message = f"""
### Error Encountered
{individual.error}

"""
            
        mutation_operator = random.choice(self.mutation_prompts)
        individual.set_operator(mutation_operator)

        final_prompt = f"""{self.task_prompt}
The current population of algorithms already evaluated (name, description, score) is:
{population_summary}

The selected solution to update is:
{description}

With code:

```python
{solution}
```


Feedback:

{feedback}

{error_message}

{mutation_operator}

"""

        return self.role_prompt + final_prompt