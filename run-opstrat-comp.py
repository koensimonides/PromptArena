import os

from llamea import Gemini_LLM, Ollama_LLM

from misc.classic_prompt import ClassicPrompt
from misc.crossover_prompt import CrossoverPrompt
from misc.mabbob import MA_BBOB_Problem
from promptarena.arena import PromptArena
from promptarena.experiment_data import ExperimentDataSet

# Sample experiment using BLADE paper data to compare operator messages
# Source data from: https://zenodo.org/records/15119985

if __name__ == "__main__": # prevents weird restarting behaviour
    experiment_name = "opstrat-comp"
    problem = MA_BBOB_Problem(dims=[5])
    budget = 200

    task_message = """The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code to minimize the function value. The code should contain an `__init__(self, budget, dim)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality can be varied.

Give an excellent and novel heuristic algorithm to solve this task."""

    operators_config = [
        ("basic", "Refine the strategy of the selected algorithm to improve it.", 1),
        ("new", "Generate a new algorithm that is different from the algorithms you have tried before.", 1),
        ("simplify", "Refine and simplify the selected algorithm to improve it.", 1),
        ("refactor", "Generate a new algorithm using core ideas from the selected algorithm.", 1),
        ("restructure", "Modify the selected algorithm by introducing a meaningful structural change that alters its overall search behaviour.", 1),
        ("correct", "Correct any mistakes in the selected algorithm.", 1),
        ("combine2", "Combine the selected algorithms by inheriting key traits from both parents.", 2),
        ("combine3", "Combine the selected algorithms by inheriting key traits from both parents.", 3),
        ("simplify2", "Simplify and recombine the selected algorithms, preserving only the most effective ideas from each.", 2),
        ("new2", "Generate a new algorithm that is different from the selected algorithms.", 2),
    ]

    def llm_gemini():
        api_key = os.getenv("GEMINI_API_KEY")
        llm_model = "gemini-2.0-flash"
        return Gemini_LLM(api_key, llm_model)

    def llm_qwen():
        llm_model = "qwen3-coder:30b"
        return Ollama_LLM(model=llm_model)

    LLM1 = llm_gemini()
    # LLM2 = llm_qwen()

    # Load log files:
    logfiles: list[str] = []
    for setup in range(5):
        logfiles = logfiles + [f"data/run-LLaMEA-{(setup + 1)}-MA_BBOB-{i}/log.jsonl" for i in range(10)]
    data = ExperimentDataSet.load_logs(logfiles)
    
    # Start at a specific index to continue experiments
    start_index = 0

    for llm in [LLM1]: # [LLM1, LLM2]:
        for operator_index in range(start_index, len(operators_config)):
            operator_id, operator_message, parent_count = operators_config[operator_index]

            # Collect log files for all experiments using different prompts
            data = ExperimentDataSet.load_logs(logfiles)

            constructor = ClassicPrompt(mutation_prompts=[operator_message], task_prompt=task_message) if parent_count == 1 else CrossoverPrompt(crossover_prompts=[operator_message], parent_count=parent_count, task_prompt=task_message)
            local_experiment_name = f"{experiment_name}-{(operator_index + 1)}"
            arena = PromptArena(problem, constructor, llm, data, budget=budget, experiment_name=local_experiment_name)
            arena.run()


