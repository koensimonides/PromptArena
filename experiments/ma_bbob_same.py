import os

from llamea import Gemini_LLM

from misc.classic_prompt import ClassicPrompt
from misc.mabbob import MA_BBOB_Problem
from promptarena.arena import PromptArena
from promptarena.experiment_data import ExperimentDataSet


# Sample experiment using BLADE paper data compared to equivalent configuration
# Source data from: https://zenodo.org/records/15119985
# Average fitness delta is expected to approach 0

expriment_name = "MA_BBOB_same"
logfiles = [f"data/run-LLaMEA-1-MA_BBOB-{i}/log.jsonl" for i in range(10)]
mutation_prompt = "Refine the strategy of the selected algorithm to improve it."
budget = 500
api_key = os.getenv("GOOGLE_API_KEY")
llm_model = "gemini-2.0-flash"
problem = MA_BBOB_Problem(dims=[5])

llm = Gemini_LLM(api_key, llm_model)
data = ExperimentDataSet.load_logs(logfiles)
constructor = ClassicPrompt(mutation_prompts=[mutation_prompt])
arena = PromptArena(problem, constructor, llm, data, budget=budget, experiment_name=expriment_name)

arena.run()
print(f"Done")

deltas = [res.fitness_delta for res in arena.results]
deltas_avg = sum(deltas) / len(deltas)
print(f"Average fitness delta: {deltas_avg}")