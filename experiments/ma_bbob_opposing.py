import os

from llamea import Gemini_LLM

from misc.classic_prompt import ClassicPrompt
from misc.mabbob import MA_BBOB_Problem
from promptarena.arena import PromptArena
from promptarena.experiment_data import ExperimentDataSet


# Experiment comparing the 5 prompt sets used in the BLADE paper
# Each prompt set will be matched with results of the other 4 
# Source data from: https://zenodo.org/records/15119985

expriment_name = "MA_BBOB_oppose"
budget = 200 # Budget per mutation prompt set
api_key = os.getenv("GOOGLE_API_KEY")
llm_model = "gemini-2.0-flash"
problem = MA_BBOB_Problem(dims=[5])

# Mutation prompts used:
mutation_prompts1 = [
    "Refine the strategy of the selected algorithm to improve it.",  # small mutation
]
mutation_prompts2 = [
    "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
]
mutation_prompts3 = [
    "Refine and simplify the selected algorithm to improve it.", #simplify
]
mutation_prompts4 = [
    "Refine the strategy of the selected solution to improve it.",  # small mutation
    "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
]
mutation_prompts5 = [
    "Refine the strategy of the selected solution to improve it.", # small mutation
    "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
    "Refine and simplify the selected algorithm to improve it.", #simplify
]
mutation_prompts_list = [mutation_prompts1, mutation_prompts2, mutation_prompts3, mutation_prompts4, mutation_prompts5]


for mpi in [2]:
    mutation_prompts = mutation_prompts_list[mpi]

    # Collect log files for all experiments using different prompts
    logfiles: list[str] = []
    for mpj in range(5):
        if mpi == mpj: 
            continue
        logfiles = logfiles + [f"data/run-LLaMEA-{(mpj + 1)}-MA_BBOB-{i}/log.jsonl" for i in range(10)]
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