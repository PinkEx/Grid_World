import numpy as np

from plot import draw_gridworld
from utils import *

if  __name__ == "__main__":
    # π, v = value_iteration()
    # π, v = truncated_policy_iteration()
    # π, v = policy_iteration()
    # π = MC_Basic_policy_evaluation()
    # v = np.zeros((n, n))
    # π, v = MC_exploring_starts()
    # π, v = MC_ε_greedy()
    # π, v = Sarsa()
    # π, v = Q_learning_on_policy()
    # π, v = Q_learning_off_policy()
    # π, v = DQN()
    π, v = A2C()
    print(v, π, sep="\n")
    draw_gridworld(π, v)