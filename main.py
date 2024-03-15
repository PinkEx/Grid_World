import numpy as np

from plot import draw_gridworld
from utils import *

if  __name__ == "__main__":
    # π, v = value_iteration()
    # π, v = policy_iteration()
    π, v = truncated_policy_iteration()
    print(v)
    print(π)
    draw_gridworld(π, v)
    

# if __name__ == "__main__":
    # π = Policy()
    # for id in range(n * n):
    #     s = id_to_state(id)
    #     p_π = {
    #         Action(dx=0, dy=0): 0.4, # O
    #         Action(dx=0, dy=-1): 0.1, # L
    #         Action(dx=0, dy=1): 0.2, # R
    #         Action(dx=-1, dy=0): 0.1, # U
    #         Action(dx=1, dy=0): 0.2, # D
    #     }
    #     π.set_action_probs(s, p_π)
    # print(policy_evaluation(π))
    # for id in range(n * n):
    #     s = id_to_state(id)
    #     print(s, ":")
    #     for a in s.action_space:
    #         print("\t", a, action_value_function(π, s, a))
        