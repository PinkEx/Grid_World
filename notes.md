## State value & Action value

$$
\begin{aligned}
v_\pi(s)
& = \mathbb{E}[G_t|S_t = s] \\
q_\pi(s, a)
& = \mathbb{E}[G_t|S_t = s, A_t = a]
\end{aligned}
$$

findings: State value is the weighted average of all action values.

### The Bellman equation(elementwise form)
$$
\begin{aligned}
v_\pi(s)
&= \sum_a \pi(a|s) q_\pi(s, a) \\
&= \sum_a \pi(a|s) \left[\sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a) v_\pi(s') \right] \\

q_\pi(s, a)
& = \sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a) v_\pi(s')
\end{aligned}
$$
where $\sum_r p(r|s, a)r$ represents immediate reward and $\gamma \sum_{s'} p(s'|s, a) v_\pi(s')$ represents discounted value of new state.

### The Bellman equation(matrix-vector form)

$$
v_\pi = r_\pi + \gamma P_\pi v_\pi
$$

usually solved in iterative solution rather than closed-form solution, where $[P_\pi]_{s, s'} \triangleq \sum_a \pi(a|s) \sum_{s'} p(s'|s, a)$.

### Bellman optimality equation(BOE, matrix-vector form)
$$
v = \max_\pi(r_\pi + \gamma P_\pi v)
$$

### Bellman optimality equation(BOE, elementwise form)
$$
\begin{aligned}
v
&= \max_\pi \sum_a \pi(a|s) q(s, a) \\
&= \max_{a \in \mathbb{A}(s)} q(s, a)
\end{aligned}
$$
and the optimality is achieved when
$$
\pi(a|s) =
\begin{cases}
1, & a = a^* \\
0, & a \neq a^* \\
\end{cases}
$$

usually solved in iterative solution rather than closed-form solution, where $a^* = \argmax_a q(s, a)$.