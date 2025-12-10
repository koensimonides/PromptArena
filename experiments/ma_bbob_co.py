import os

from llamea import Gemini_LLM

from misc.crossover_prompt import CrossoverPrompt
from misc.mabbob import MA_BBOB_Problem
from promptarena.arena import PromptArena
from promptarena.experiment_data import ExperimentDataSet


# Experiment using BLADE paper data comparing equivalent configuration with crossover instead of mutation
# Source data from: https://zenodo.org/records/15119985

expriment_name = "MA_BBOB_co"
logfiles = [f"data/run-LLaMEA-1-MA_BBOB-{i}/log.jsonl" for i in range(10)]
budget = 200
api_key = os.getenv("GOOGLE_API_KEY")
llm_model = "gemini-2.0-flash"
problem = MA_BBOB_Problem(dims=[5])

crossover_prompts = [
    "Combine the algorithmic structures of the following two optimization methods by simulating a natural genetic crossover, producing a hybrid metaheuristic that inherits key traits from both parents.",
]

llm = Gemini_LLM(api_key, llm_model)
data = ExperimentDataSet.load_logs(logfiles)
constructor = CrossoverPrompt(crossover_prompts=crossover_prompts)
arena = PromptArena(problem, constructor, llm, data, budget=budget, experiment_name=expriment_name)

arena.run()
print(f"Done")

fitness_deltas = [res.fitness_delta for res in arena.results]
fitness_deltas_avg = sum(fitness_deltas) / len(fitness_deltas)
success_deltas = [((1 if res.fitness_delta > 0 else -1) if res.fitness_delta != 0 else 0) for res in arena.results]
success_deltas_avg = sum(success_deltas) / len(success_deltas)
print(f"Average fitness delta: {fitness_deltas_avg}")
print(f"Average success delta: {success_deltas_avg}")