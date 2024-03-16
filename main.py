import numpy as np

from plot import draw_gridworld
from utils import *

if  __name__ == "__main__":
    # π, v = value_iteration()
    # π, v = truncated_policy_iteration()
    # π, v = policy_iteration()
    # π = MC_Basic_policy_evaluation()
    # π = MC_exploring_starts()
    π = MC_ε_greedy()
    # print(v)
    print(π)
    v = np.zeros((n, n))
    draw_gridworld(π, v)