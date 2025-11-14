from iohblade.problems import MA_BBOB
from llamea import Solution


# Wrapper for MMA_BBOB problem defined in IOHBlade 
class MA_BBOB_Problem(object):
    def __init__(
        self,
        dims=[2, 5],
    ):
        self.problem = MA_BBOB(dims) # seeds?


    def __call__(self, individual: Solution) -> Solution:
        return self.problem.evaluate(individual, test=False, ioh_dir="")