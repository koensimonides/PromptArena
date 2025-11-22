import os

from llamea import Gemini_LLM

from misc.classic_prompt import ClassicPrompt
from misc.mabbob import MA_BBOB_Problem
from promptarena.arena import PromptArena
from promptarena.experiment_data import ExperimentDataSet


# Sample experiment using BLADE paper data comparing different mutation prompt to baseline prompt, sampling parent fitness vs fitness delta
# Source data from: https://zenodo.org/records/15119985
# Average fitness delta is expected to approach 0

expriment_name = "MA_BBOB_fdf"
logfiles = [f"data/run-LLaMEA-1-MA_BBOB-{i}/log.jsonl" for i in range(10)]
budget = 200
api_key = os.getenv("GOOGLE_API_KEY")
llm_model = "gemini-2.0-flash"
problem = MA_BBOB_Problem(dims=[5])

# Mutation prompts used:
mutation_prompts1 = [
    "Refine the strategy of the selected algorithm to improve it.", #baseline
]
mutation_prompts2 = [
    "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
]
mutation_prompts3 = [
    "Refine and simplify the selected algorithm to improve it.", #simplify
]
mutation_prompts4 = [
    "Generate a new algorithm using the core ideas from the selected algorithm", #refactor 1
]
mutation_prompts5 = [
    "Generate a new algorithm by combining concepts from the algorithms you have tried before.", #refactor +
]
mutation_prompts_list = [mutation_prompts1, mutation_prompts2, mutation_prompts3, mutation_prompts4, mutation_prompts5]

for mpi in range(5):
    mutation_prompts = mutation_prompts_list[mpi]

    # Collect log files for all experiments using different prompts
    data = ExperimentDataSet.load_logs(logfiles)


    llm = Gemini_LLM(api_key, llm_model)
    constructor = ClassicPrompt(mutation_prompts=mutation_prompts)
    local_experiment_name = f"{expriment_name}-{(mpi + 1)}"
    arena = PromptArena(problem, constructor, llm, data, budget=budget, experiment_name=local_experiment_name)
    arena.run()

    
    fitness_deltas = [res.fitness_delta for res in arena.results]
    fitness_deltas_avg = sum(fitness_deltas) / len(fitness_deltas)
    success_deltas = [((1 if res.fitness_delta > 0 else -1) if res.fitness_delta != 0 else 0) for res in arena.results]
    success_deltas_avg = sum(success_deltas) / len(success_deltas)
    print(f"Average fitness delta: {fitness_deltas_avg}")
    print(f"Average success delta: {success_deltas_avg}")

print(f"Done")