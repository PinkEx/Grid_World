n = 6
r_target = 1 # encouragement
r_bound = -1 # punishment
r_forbid = -1 # punishment
γ = 0.9 # discount_rate
θ = 1e-10 # error threshold
env = [
    [".", ".", ".", ".", ".", "."],
    [".", "E", "*", ".", "*", "."],
    ["*", "*", ".", ".", ".", "."],
    [".", ".", ".", ".", "*", "*"],
    [".", "*", ".", "*", "E", "."],
    [".", ".", ".", ".", ".", "."],
]