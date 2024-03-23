from collections import defaultdict
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

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
    return env[x][y].isdigit() and int(env[x][y]) >= 1

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
    x, y = s.x + a.dx, s.y + a.dy
    if outside(x, y): return r_bound
    if forbid(x, y): return r_forbid
    r = r_stay if (a.dx, a.dy) == (0, 0) else r_other
    if target(x, y):
        r += float(env[x][y]) # r_target
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

def MC_exploring_starts(num_episode: int = 10000) -> Tuple[Policy, np.array]:
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
    v = np.zeros((n, n))
    for id in range(n * n):
        s = id_to_state(id)
        for a in s.action_space:
            lst = g_samples[(s, a)]
            v[s.x][s.y] += π[s][a] * (sum(lst) / len(lst))
    return π, v

# a balance between exploitation and exploration
def MC_ε_greedy(num_episode: int = 10000, ε: float = 0.2) -> Tuple[Policy, np.array]:
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
    v = np.zeros((n, n))
    for id in range(n * n):
        s = id_to_state(id)
        for a in s.action_space:
            lst = g_samples[(s, a)]
            v[s.x][s.y] += π[s][a] * (sum(lst) / len(lst))
    return π, v

# Sarsa(State-Action-Reward-State-Action-Reward), on-policy
def Sarsa(num_episode: int = 500, ε: float = 0.1, α: float = 0.1) -> Tuple[Policy, np.array]:
    π = initial_guess_policy()
    q_value = defaultdict(float)
    while num_episode > 0:
        num_episode -= 1
        # start from a certain state -> not all states have optimal policy
        s = id_to_state(12)
        a = random.choices(
            s.action_space,
            weights=[
                π[s][a] for a in s.action_space
            ],
            k=1
        )[0]
        length = 0
        while True:
            length += 1
            r = float(action_reward(s, a))
            s_prime = state_transition(s, a)
            a_prime = random.choices(
                s_prime.action_space,
                weights=[
                    π[s_prime][a_prime] for a_prime in s_prime.action_space
                ],
                k=1
            )[0]
            # update q-value
            new_value = q_value[(s, a)] - α * (q_value[(s, a)] - (r + γ * q_value[(s_prime, a_prime)]))
            q_value[(s, a)] = new_value
            # update policy
            a_stars, max_q = [], -inf
            for a in s.action_space:
                value = q_value[(s, a)]
                if max_q < value:
                    max_q = value
                    a_stars = []
                if max_q == value:
                    a_stars.append(a)
            p_π = {}
            p_minor = ε / len(s.action_space)
            for a in s.action_space:
                p_π[a] = p_minor
            for a_star in a_stars:
                p_π[a_star] = (1.0 - (len(s.action_space) - len(a_stars)) * p_minor) / len(a_stars)
            π.set_action_probs(s, p_π)
            if target(s.x, s.y): break
            s, a = s_prime, a_prime
        # print(num_episode, length)
    v = np.zeros((n, n))
    for id in range(n * n):
        s = id_to_state(id)
        for a in s.action_space:
            v[s.x][s.y] += π[s][a] * q_value[(s, a)]
    return π, v


# Q-learning, on-policy version
def Q_learning_on_policy(num_episode: int = 500, ε: float = 0.1, α: float = 0.1) -> Tuple[Policy, np.array]:
    π = initial_guess_policy()
    q_value = defaultdict(float)
    while num_episode > 0:
        num_episode -= 1
        # start from a certain state -> not all states have optimal policy
        s = id_to_state(12)
        a = random.choices(
            s.action_space,
            weights=[
                π[s][a] for a in s.action_space
            ],
            k=1
        )[0]
        length = 0
        while True:
            length += 1
            r = float(action_reward(s, a))
            s_prime = state_transition(s, a)
            # update q-value
            max_q = -inf
            for a_prime in s_prime.action_space:
                max_q = max(max_q, q_value[(s_prime, a_prime)])
            new_value = q_value[(s, a)] - α * (q_value[(s, a)] - (r + γ * max_q))
            q_value[(s, a)] = new_value
            # update policy
            a_stars, max_q = [], -inf
            for a in s.action_space:
                value = q_value[(s, a)]
                if max_q < value:
                    max_q = value
                    a_stars = []
                if max_q == value:
                    a_stars.append(a)
            p_π = {}
            p_minor = ε / len(s.action_space)
            for a in s.action_space:
                p_π[a] = p_minor
            for a_star in a_stars:
                p_π[a_star] = (1.0 - (len(s.action_space) - len(a_stars)) * p_minor) / len(a_stars)
            π.set_action_probs(s, p_π)
            if target(s.x, s.y): break
            s = s_prime
            a = random.choices(
                s_prime.action_space,
                weights=[
                    π[s_prime][a_prime] for a_prime in s_prime.action_space
                ],
                k=1
            )[0]
        # print(num_episode, length)
    v = np.zeros((n, n))
    for id in range(n * n):
        s = id_to_state(id)
        for a in s.action_space:
            v[s.x][s.y] += π[s][a] * q_value[(s, a)]
    return π, v

# Q-learning, off-policy version
def Q_learning_off_policy(num_episode: int = 500, ε: float = 0.1, α: float = 0.1) -> Tuple[Policy, np.array]:
    π = initial_guess_policy()
    q_value = defaultdict(float)
    while num_episode > 0:
        num_episode -= 1
        # start from a certain state -> not all states have optimal policy
        sa_list = []
        s = id_to_state(12)
        a = random.choices(
            s.action_space,
            weights=[
                π[s][a] for a in s.action_space
            ],
            k=1
        )[0]
        while not target(s.x, s.y):
            sa_list.append((s, a))
            s_prime = state_transition(s, a)
            a_prime = random.choices(
                s_prime.action_space,
                weights=[
                    π[s_prime][a_prime] for a_prime in s_prime.action_space
                ],
                k=1
            )[0]
            s, a = s_prime, a_prime
        # print(num_episode, len(sa_list))
        for s, a in sa_list:
            r = float(action_reward(s, a))
            s_prime = state_transition(s, a)
            # update q-value
            max_q = -inf
            for a_prime in s_prime.action_space:
                max_q = max(max_q, q_value[(s_prime, a_prime)])
            new_value = q_value[(s, a)] - α * (q_value[(s, a)] - (r + γ * max_q))
            q_value[(s, a)] = new_value
            # update policy
            a_stars, max_q = [], -inf
            for a in s.action_space:
                value = q_value[(s, a)]
                if max_q < value:
                    max_q = value
                    a_stars = []
                if max_q == value:
                    a_stars.append(a)
            p_π = {}
            p_minor = ε / len(s.action_space)
            for a in s.action_space:
                p_π[a] = p_minor
            for a_star in a_stars:
                p_π[a_star] = (1.0 - (len(s.action_space) - len(a_stars)) * p_minor) / len(a_stars)
            π.set_action_probs(s, p_π)
    v = np.zeros((n, n))
    for id in range(n * n):
        s = id_to_state(id)
        for a in s.action_space:
            v[s.x][s.y] += π[s][a] * q_value[(s, a)]
    return π, v

class NetW(nn.Module):
    def __init__(self):
        super(NetW, self).__init__()
        self.fc1 = nn.Linear(4, 200)
        self.fc2 = nn.Linear(200, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.sigmoid(self.fc2(x))
        return x

def DQN(iter_times: int = 5000, batch_size: int = 10):
    W = NetW()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(W.parameters(), lr=0.01)
    input_list = []
    output_list = []
    
    def update_network(iter_id: int, epochs: int = 20):
        print(iter_id)
        for epoch in range(epochs):
            optimizer.zero_grad()
            inputs = torch.tensor(input_list, dtype=torch.float32)
            outputs = torch.tensor(output_list, dtype=torch.float32)
            outputs_pred = W(inputs)
            loss = criterion(outputs_pred, outputs)
            loss.backward()
            optimizer.step()
            print(f"  Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

    experience_list: List[Tuple[State, Action]] = []
    for id in range(n * n):
        s = id_to_state(id)
        if forbid(s.x, s.y) or target(s.x, s.y): continue
        for a in s.action_space:
            experience_list.append((s, a))
    random.shuffle(experience_list)

    while iter_times > 0:
        iter_times -= 1
        # print(iter_times)
        sample_list = random.choices(
            experience_list,
            k=batch_size
        )
        for s, a in sample_list:
            r = float(action_reward(s, a))
            s_prime = state_transition(s, a)
            max_q = -inf
            for a_prime in s_prime.action_space:
                in_tensor = torch.tensor([[s_prime.x, s_prime.y, a_prime.dx, a_prime.dy]], dtype=torch.float32)
                out_tensor = W(in_tensor)
                max_q = max(max_q, out_tensor[0])
            v = r + γ * max_q
            input_list.append([s.x, s.y, a.dx, a.dy])
            output_list.append([v])
        if iter_times % 100 == 0:
            update_network(iter_times)
            input_list.clear()
            output_list.clear()

    π = initial_guess_policy()
    v = np.zeros((n, n))
    for id in range(n * n):
        s = id_to_state(id)
        a_stars, max_q = [], -inf
        for a in s.action_space:
            in_tensor = torch.tensor([[s.x, s.y, a.dx, a.dy]], dtype=torch.float32)
            out_tensor = W(in_tensor)
            value = float(out_tensor[0])
            if max_q < value:
                max_q = value
                a_stars = []
            if max_q == value:
                a_stars.append(a)
            p_π = {}
            for a in s.action_space:
                p_π[a] = 0.0
            for a_star in a_stars:
                p_π[a_star] = 1.0 / len(a_stars)
            π.set_action_probs(s, p_π)
        for a in s.action_space:
            in_tensor = torch.tensor([[s.x, s.y, a.dx, a.dy]], dtype=torch.float32)
            out_tensor = W(in_tensor)
            value = float(out_tensor[0])
            v[s.x][s.y] += π[s][a] * value
    return π, v

class NetValue(nn.Module):
    def __init__(self) -> None:
        super(NetValue, self).__init__()
        self.fc1 = nn.Linear(2, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NetPolicy(nn.Module):
    def __init__(self) -> None:
        super(NetPolicy, self).__init__()
        self.fc1 = nn.Linear(2, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        prob = F.softmax(x, dim=-1)
        return prob

def get_action(π: NetPolicy, s: State):
    probs = π(torch.tensor([s.x, s.y], dtype=torch.float32))
    m = Categorical(probs)
    a_id = m.sample()
    logp = m.log_prob(a_id)
    a = Action()
    if a_id == 0: a.dx, a.dy = 0, 0
    elif a_id == 1: a.dx, a.dy = 0, -1
    elif a_id == 2: a.dx, a.dy = 0, 1
    elif a_id == 3: a.dx, a.dy = -1, 0
    elif a_id == 4: a.dx, a.dy = 1, 0
    # print("select:", s, a, f"(a_id: {a_id})")
    # print(probs, logp)
    return a, logp

def A2C(num_episode: int = 10000, max_len: int = 10):
    π, v = NetPolicy(), NetValue()
    π_opt = optim.Adam(π.parameters(), lr=3e-4)
    v_opt = optim.Adam(v.parameters(), lr=6e-4)
    while num_episode > 0:
        num_episode -= 1
        id = random.randint(0, n * n - 1)
        s = id_to_state(id)
        while outside(s.x, s.y) or forbid(s.x, s.y):
            id = random.randint(0, n * n - 1)
            s = id_to_state(id)
        a, logp = get_action(π, s)
        lth = 0
        while lth < max_len:
            lth += 1
            r = float(action_reward(s, a))
            s_prime = state_transition(s, a)
            v_s = v(torch.tensor([s.x, s.y], dtype=torch.float32))
            v_s_prime = v(torch.tensor([s_prime.x, s_prime.y], dtype=torch.float32))
            
            # TD error - using advantage function
            δ = r + γ * v_s_prime - v_s
            
            # print(f" <{s.x}, {s.y}> + <{a.dx}, {a.dy}> -> <{s_prime.x}, {s_prime.y}>: δ = {δ}")
            # print(f"v_s = {v_s}, v_s' = {v_s_prime}")
            
            # Critic (value update)
            critic_loss = F.mse_loss(r + γ * v_s_prime, v_s)
            # print(f"critic_loss: {critic_loss}")
            v_opt.zero_grad()
            critic_loss.backward()
            v_opt.step()
            
            # Actor (policy update)
            actor_loss = (-δ.detach() * logp).mean()
            # print(f"actor_loss: {actor_loss}\n")
            π_opt.zero_grad()
            actor_loss.backward()
            π_opt.step()

            s = s_prime
            a, logp = get_action(π, s)

        if num_episode % 10 == 0:        
            print(num_episode)
            print("V:")
            for x in range(5):
                for y in range(5):
                    print(round(float(v(torch.tensor([x, y], dtype=torch.float32))), 2), end=" ")
                print()
    
    _v = np.zeros((n, n))
    _π = initial_guess_policy()
    for x in range(n):
        for y in range(n):
            _v[x][y] = float(v(torch.tensor([x, y], dtype=torch.float32)))
            probs = π(torch.tensor([x, y], dtype=torch.float32))
            s = State(x, y)
            _π.set_action_probs(
                s, {a: float(probs[a.__hash__()]) for a in s.action_space}
            )
    return _π, _v