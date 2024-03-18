import numpy as np
import matplotlib.pyplot as plt

from constant import *
from utils import forbid, target

from action import Action
from policy import Policy
from state import State

def draw_gridworld(π: Policy, v: np.array):
    fig, ax = plt.subplots()

    # Draw gridlines
    for i in range(n + 1):
        ax.axhline(i, color='gray', lw=2)
    for j in range(n + 1):
        ax.axvline(j, color='gray', lw=2)

    # # Draw patches
    for i in range(n):
        for j in range(n):
            s = State(i, j)
            x, y = j, n - i - 1
            color = "yellow" if env[i][j] == "*" else \
                    "skyblue" if env[i][j] == "1" else \
                    "cornflowerblue" if env[i][j] == "2" else \
                    "white"
            ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='black'))
            if forbid(i, j) or target(i, j): continue
            ax.text(x + 0.5, y + 0.5, str(round(v[i][j], 2)), ha='center', va='center')
            for a in s.action_space:
                if π[s][a] == 0: continue
                l = 0.3 * π[s][a]
                if a == Action(0, 0):
                    ax.scatter(x + 0.5, y + 0.5, color="pink", s=100 * l)
                elif a == Action(-1, 0):
                    ax.arrow(x + 0.5, y + 0.5, 0, l, head_width=0.1, head_length=0.1, fc="pink", ec="pink")
                elif a == Action(1, 0):
                    ax.arrow(x + 0.5, y + 0.5, 0, -l, head_width=0.1, head_length=0.1, fc="pink", ec="pink")
                elif a == Action(0, 1):
                    ax.arrow(x + 0.5, y + 0.5, l, 0, head_width=0.1, head_length=0.1, fc="pink", ec="pink")
                elif a == Action(0, -1):
                    ax.arrow(x + 0.5, y + 0.5, -l, 0, head_width=0.1, head_length=0.1, fc="pink", ec="pink")

    # Set aspect of the plot to equal
    ax.set_aspect('equal')

    # Remove axes
    ax.axis('off')

    fig = plt.gcf()
    fig.canvas.manager.set_window_title('Grid World')

    plt.show()
