import os

from llamea import Gemini_LLM

from misc.classic_prompt import ClassicPrompt
from misc.crossover_prompt import CrossoverPrompt
from misc.mabbob import MA_BBOB_Problem
from promptarena.arena import PromptArena
from promptarena.experiment_data import ExperimentDataSet


# Sample experiment using BLADE paper data comparing different mutation prompts, sampling parent fitness vs fitness delta
# Source data from: https://zenodo.org/records/15119985
# Average fitness delta is expected to approach 0

expriment_name = "MA_BBOB_fdf2"
budget = 200
api_key = os.getenv("GOOGLE_API_KEY")
llm_model = "gemini-2.0-flash"
problem = MA_BBOB_Problem(dims=[5])

# Load log files:
logfiles: list[str] = []
for setup in range(5):
    logfiles = logfiles + [f"data/run-LLaMEA-{(setup + 1)}-MA_BBOB-{i}/log.jsonl" for i in range(10)]
data = ExperimentDataSet.load_logs(logfiles)

# Prompts
prompts_lists: list[tuple[list[str], int]] = [
    ([ #basic
        "Refine the strategy of the selected algorithm to improve it.", 
    ], 1),
    ([ #new random solution
        "Generate a new algorithm that is different from the algorithms you have tried before.", 
    ], 1),
    ([ #simplify
        "Refine and simplify the selected algorithm to improve it.", 
    ], 1),
    ([ #refactor 1
        "Generate a new algorithm using the core ideas from the selected algorithm.", 
    ], 1),
    ([ #refactor +
        "Generate a new algorithm by combining concepts from the algorithms you have tried before.",
    ], 1),
    ([ #single structural change
        "Modify the selected algorithm by introducing a meaningful structural change that alters its overall search behavior.",
    ], 1),
    ([ # combine 2
        "Combine the algorithmic structures of the following two optimization methods by simulating a natural genetic crossover, producing a hybrid metaheuristic that inherits key traits from both parents.",
    ], 2), 
    ([ # combine 3
        "Combine the algorithmic structures of the following two optimization methods by simulating a natural genetic crossover, producing a hybrid metaheuristic that inherits key traits from both parents.",
    ], 3), 
    ([ # crossover simplify 2
        "Simplify and recombine the selected algorithms, preserving only the most effective ideas from each.",
    ], 2), 
    ([ # crossover new random 2
        "Generate a new algorithm that is different from the selected two optimization methods.",
    ], 2), 
]

# Fix the method
method = 0

for mpi in range(5):
    prompts, parents = prompts_lists[mpi]

    # Collect log files for all experiments using different prompts
    data = ExperimentDataSet.load_logs(logfiles)


    llm = Gemini_LLM(api_key, llm_model)
    constructor = ClassicPrompt(mutation_prompts=prompts) if parents == 1 else CrossoverPrompt(crossover_prompts=prompts, parent_count=parents)
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