from collections import defaultdict
import random
from typing import List, Tuple

import numpy as np

from constant import *

from action import Action
from policy import Policy
from reward import Reward
from state import State

def id_to_state(id: int) -> State:
    return State(id // n, id % n)

def outside(x, y) -> bool:
    return x < 0 or x >= n or y < 0 or y >= n

def forbid(x, y):
    return env[x][y] == "*"

def target(x, y):
    return env[x][y] == "E"

def initial_guess_policy() -> Policy:
    π = Policy()
    for id in range(n * n):
        s = id_to_state(id)
        p_π = {}
        for a in s.action_space:
            p_π[a] = 1.0 / len(s.action_space)
        π.set_action_probs(s, p_π)
    return π

# deterministic case
def state_transition(s: State, a: Action) -> State:
    x, y = s.x + a.dx, s.y + a.dy
    if outside(x, y) or forbid(x, y): x, y = s.x, s.y
    return State(
        x=x, y=y
    )

# human-machine interface
def action_reward(s: State, a: Action) -> Reward:
    x, y, r = s.x + a.dx, s.y + a.dy, 0.0
    if outside(x, y): r = r_bound
    elif forbid(x, y): r = r_forbid
    elif target(x, y): r = r_target
    return Reward(value=r)

def reward_function(s: State, π: Policy) -> Reward:
    r = Reward(0)
    for a in s.action_space:
        r += action_reward(s, a) * π.get_action_prob(s, a)
    return r

# transition probability
def transition_prob(s: State, s_prime: State, π: Policy) -> float:
    prob = 0
    for a in s.action_space:
        if state_transition(s, a) == s_prime:
            prob += π.get_action_prob(s, a)
    return prob

# to find out the corresponding STATE VALUEs under certain policy
# v_prime = r_π + γP_π * v, k -> ∞, v_prime -> v_π
# dimension: (n * n, 1) = (n * n, 1) + c × (n * n, n * n) × (n * n, 1)
def policy_evaluation(π: Policy) -> np.array:
    v = np.zeros((n * n, 1))
    r_π = np.reshape(
        np.array(
            [float(reward_function(State(x, y), π)) for x in range(n) for y in range(n)]
        ),
        (n * n, 1)
    )
    P_π = np.array(
        [[transition_prob(id_to_state(s), id_to_state(s_prime), π) for s_prime in range(n * n)] for s in range(n * n)]
    )
    while True:
        v_prime = r_π + γ * P_π @ v
        err = np.linalg.norm(v_prime - v)
        if err < θ: break
        v = v_prime.copy()
    return np.reshape(v, (n, n))

# action_value[s, a] = immediate_reward[s, a] + new_state_value(discounted)[s, a]
def action_value_function(π: Policy, s: State, a: Action) -> float:
    v_π = policy_evaluation(π)
    s_prime = state_transition(s, a)
    r = action_reward(s, a)
    return float(r + γ * v_π[s_prime.x][s_prime.y])

# To solve the Bellman Optimality Equation(BOE), value iteration|truncated policy iteration/policy iteration
def value_iteration() -> Tuple[Policy, np.array]:
    π = initial_guess_policy()
    # (one-time) policy evaluation
    v = policy_evaluation(π)
    # policy improvement
    while True:
        v_prime = v.copy()
        for id in range(n * n):
            s = id_to_state(id)
            q = {}
            for a in s.action_space:
                s_prime = state_transition(s, a)
                q[a] = float(action_reward(s, a) + γ * v[s_prime.x][s_prime.y])
            max_q = max(q.values())
            a_stars = [a for a, q in q.items() if q == max_q]
            # policy update
            p_π = {}
            for a in s.action_space:
                p_π[a] = 0.0
            for a_star in a_stars:
                p_π[a_star] = 1.0 / len(a_stars)
            π.set_action_probs(s, p_π)
            # value update
            v_prime[s.x][s.y] = q[a_star]
        err = np.linalg.norm(v_prime - v)
        if err < θ: break
        v = v_prime.copy()
    return π, v
    
def policy_iteration() -> Tuple[Policy, np.array]:
    π = initial_guess_policy()
    while True:
        # policy evaluation
        v = policy_evaluation(π)
        # policy improvement
        flag = False
        for id in range(n * n):
            s = id_to_state(id)
            q = {}
            for a in s.action_space:
                s_prime = state_transition(s, a)
                q[a] = float(action_reward(s, a) + γ * v[s_prime.x][s_prime.y])
            max_q = max(q.values())
            a_stars = [a for a, q in q.items() if q == max_q]
            p_π = {}
            for a in s.action_space:
                p_π[a] = 0.0
            for a_star in a_stars:
                p_π[a_star] = 1.0 / len(a_stars)
            if p_π != π[s]: flag = True
            π.set_action_probs(s, p_π)
        if not flag: break
    return π, v

def truncated_policy_iteration(trunc_times: int = 10) -> Tuple[Policy, np.array]:
    π = initial_guess_policy()
    count = 0
    while count < trunc_times:
        count += 1
        # policy evaluation
        v = policy_evaluation(π)
        # policy improvement
        for id in range(n * n):
            s = id_to_state(id)
            q = {}
            for a in s.action_space:
                s_prime = state_transition(s, a)
                q[a] = float(action_reward(s, a) + γ * v[s_prime.x][s_prime.y])
            max_q = max(q.values())
            a_stars = [a for a, q in q.items() if q == max_q]
            p_π = {}
            for a in s.action_space:
                p_π[a] = 0.0
            for a_star in a_stars:
                p_π[a_star] = 1.0 / len(a_stars)
            π.set_action_probs(s, p_π)
    return π, v

# low efficiency
def MC_search(π: Policy, s: State, l: int) -> float:
    g = []
    if l < max_len:
        for a in π[s].keys():
            if π[s][a] == 0.0: continue
            s_prime = state_transition(s, a)
            g.append(float(action_reward(s, a) + γ * MC_search(π, s_prime, l + 1)))
    return sum(g) / len(g) if len(g) > 0 else 0

def collect_episodes(π: Policy, s: State, a: Action) -> float:
    s_prime = state_transition(s, a)
    r = action_reward(s, a)
    return float(r + MC_search(π, s_prime, 0))

def MC_Basic_policy_evaluation() -> Policy:
    π = initial_guess_policy()
    while True:
        flag = False
        for id in range(n * n):
            s = id_to_state(id)
            # policy evaluation
            q = {}
            for a in s.action_space:
                q[a] = collect_episodes(π, s, a)
            # policy improvement
            max_q = max(q.values())
            a_stars = [a for a, q in q.items() if q == max_q]
            p_π = {}
            for a in s.action_space:
                p_π[a] = 0.0
            for a_star in a_stars:
                p_π[a_star] = 1.0 / len(a_stars)
            if π[s] != p_π: flag = True
            π.set_action_probs(s, p_π)
        if not flag: break
    return π

def episode_gen(π: Policy, s: State, a: Action) -> List[Tuple[State, Action]]:
    episode = [(s, a)]
    while len(episode) < max_len:
        s, a = episode[-1]
        s_prime = state_transition(s, a)
        a_prime = random.choices(
            s_prime.action_space,
            weights=[
                π[s_prime][a] for a in s_prime.action_space
            ],
            k=1
        )[0]
        episode.append((s_prime, a_prime))
    return episode

def MC_exploring_starts(num_episode: int = 10000) -> Policy:
    π = initial_guess_policy()
    g_samples = defaultdict(list)
    while num_episode > 0:
        num_episode -= 1
        id = random.randint(0, n * n - 1)
        s = id_to_state(id)
        a = random.choice(s.action_space)
        episode = episode_gen(π, s, a)
        g = 0.0
        for t in range(-1, -(max_len + 1), -1):
            s, a = episode[t]
            g = γ * g + float(action_reward(s, a))
            if (s, a) in episode[:t]: continue
            lst = g_samples[(s, a)]
            lst.append(g)
            g_samples[(s, a)] = lst
            a_stars, max_q = [], 0.0
            for a in s.action_space:
                lst = g_samples[(s, a)]
                if len(lst) == 0: continue
                if a_stars == [] or max_q == sum(lst) / len(lst):
                    max_q = sum(lst) / len(lst)
                    a_stars.append(a)
                elif max_q < sum(lst) / len(lst):
                    max_q = sum(lst) / len(lst)
                    a_stars = [a]
            p_π = {}
            for a in s.action_space:
                p_π[a] = 0.0
            for a_star in a_stars:
                p_π[a_star] = 1.0 / len(a_stars)
            π.set_action_probs(s, p_π)
    return π

# a balance between exploitation and exploration
def MC_ε_greedy(num_episode: int = 10000, ε: float = 0.1):
    π = initial_guess_policy()
    g_samples = defaultdict(list)
    while num_episode > 0:
        num_episode -= 1
        id = random.randint(0, n * n - 1)
        s = id_to_state(id)
        a = random.choice(s.action_space)
        episode = episode_gen(π, s, a)
        g = 0.0
        for t in range(-1, -(max_len + 1), -1):
            s, a = episode[t]
            g = γ * g + float(action_reward(s, a))
            if (s, a) in episode[:t]: continue
            lst = g_samples[(s, a)]
            lst.append(g)
            g_samples[(s, a)] = lst
            a_stars, max_q = [], 0.0
            for a in s.action_space:
                lst = g_samples[(s, a)]
                if len(lst) == 0: continue
                if a_stars == [] or max_q == sum(lst) / len(lst):
                    max_q = sum(lst) / len(lst)
                    a_stars.append(a)
                elif max_q < sum(lst) / len(lst):
                    max_q = sum(lst) / len(lst)
                    a_stars = [a]
            p_π = {}
            p_minor = ε / len(s.action_space)
            for a in s.action_space:
                p_π[a] = p_minor
            for a_star in a_stars:
                p_π[a_star] = (1.0 - (len(s.action_space) - len(a_stars)) * p_minor) / len(a_stars)
            π.set_action_probs(s, p_π)
    return π