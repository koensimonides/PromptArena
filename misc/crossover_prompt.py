import random

from llamea import Solution

from promptarena.prompt_constructor import PromptConstructor


# Takes inspiration from LLaMEA crossover prompt constructor function
# https://github.com/XAI-liacs/LLaMEA/blob/e4eafe7d684a48839fef65fb7bc64be42a57e8cd/llamea/llamea.py#L423
class CrossoverPrompt(PromptConstructor):

    def __init__(
            self,
            task_prompt = "",
            role_prompt = "",
            crossover_prompts = None,
            parent_count: int = 2
            ):
        super().__init__(parent_count)

        self.task_prompt = task_prompt
        self.role_prompt = role_prompt
        if role_prompt == "":
            self.role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."

        self.crossover_prompts = crossover_prompts
        if crossover_prompts == None:
            self.crossover_prompts = [
                "Combine the algorithmic structures of the following two optimization methods by simulating a natural genetic crossover, producing a hybrid metaheuristic that inherits key traits from both parents.",
            ]
   
    def make_prompt(self, parents: list[Solution], population: list[Solution]) -> str:
        population_summary = "\n".join([ind.get_summary() for ind in population])
            
        mutation_operator = random.choice(self.crossover_prompts)

        parent_infos = ""

        for parent in parents:
            parent_infos += f"""
{self.make_parent_info(parent)}
"""

        final_prompt = f"""{self.task_prompt}
The current population of algorithms already evaluated (name, description, score) is:
{population_summary}

The selected algorithms are:
{parent_infos}

{mutation_operator}
"""

        return self.role_prompt + final_prompt
    
    def make_parent_info(self, parent: Solution) -> str:
        error_message = ""
        if parent.error:
            error_message = f"""
### Error Encountered
{parent.error}

"""
            
        return f"""{parent.description}

With code:

```python
{parent.code}
```


Feedback:

{parent.feedback}

{error_message}
"""