import random

from llamea import Solution

from promptarena.prompt_constructor import PromptConstructor


# Roughly copied from LLaMEA prompt constructor function
# https://github.com/XAI-liacs/LLaMEA/blob/9209b25b1fd11744f78a19df5873aad23a4407cf/llamea/llamea.py#L453
class ClassicPrompt(PromptConstructor):

    def __init__(
            self,
            task_prompt = "",
            role_prompt = "",
            mutation_prompts = None
            ):
        super().__init__(1)

        self.task_prompt = task_prompt
        self.role_prompt = role_prompt
        if role_prompt == "":
            self.role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."

        self.mutation_prompts = mutation_prompts
        if mutation_prompts == None:
            self.mutation_prompts = [
                "Refine the strategy of the selected solution to improve it.",
            ]

    def make_prompt(self, parents: list[Solution], population: list[Solution]) -> str:
        individual: Solution = parents[0]
        
        population_summary = "\n".join([ind.get_summary() for ind in population])
        error_message = ""
        if individual.error:
            error_message = f"""
### Error Encountered
{individual.error}

"""
            
        mutation_operator = random.choice(self.mutation_prompts)

        final_prompt = f"""{self.task_prompt}
The current population of algorithms already evaluated (name, description, score) is:
{population_summary}

The selected solution to update is:
{individual.description}

With code:

```python
{individual.code}
```


Feedback:

{individual.feedback}

{error_message}

{mutation_operator}
"""

        return self.role_prompt + final_prompt