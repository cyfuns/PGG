import numpy as np

def find_neighbors(node, L):
    x, y = node
    return [
        ((x + 1) % L, y),  # Right neighbor
        ((x - 1) % L, y),  # Left neighbor
        (x, (y + 1) % L),  # Down neighbor
        (x, (y - 1) % L)  # Up neighbor
    ]

def calculate_payoff(node, cooperator_grid, L, r):
    neighbors = find_neighbors(node, L)
    node_type = cooperator_grid[node]
    total_payoff = 0
    for neighbor in neighbors:
        neighbor_type = cooperator_grid[neighbor]
        # Calculate mutual benefit from current neighbor and the node itself
        neighbor_neighbors = find_neighbors(neighbor, L)
        cooperations = sum(cooperator_grid[n] == 0 for n in neighbor_neighbors)
        if neighbor_type == 0:  # If neighbor is a cooperator, include itself in cooperation count
            cooperations += 1
        total_payoff += cooperations * r / 5 - (1 if neighbor_type == 0 else 0)
    # Add the payoff for the node being a center
    cooperations_center = sum(cooperator_grid[n] == 0 for n in neighbors)
    if node_type == 0:  # If node is a cooperator, include itself
        cooperations_center += 1
    total_payoff += cooperations_center * r / 5 - (1 if node_type == 0 else 0)
    return total_payoff

L = 1000
r = 5
K = 0.1
cooperator_grid = np.random.choice([0, 1], size=(L, L), p=[0.5, 0.5])  # 0 is cooperator, 1 is defector

iterations = 100000000  # Use a reasonable number for practical runtimes
output_frequency = 10000  # Output frequency for monitoring

for iteration in range(iterations):
    x, y = np.random.randint(0, L, 2)
    node_x = (x, y)
    node_y = find_neighbors(node_x, L)[np.random.randint(4)]  # Pick a random neighbor

    payoff_x = calculate_payoff(node_x, cooperator_grid, L, r)
    payoff_y = calculate_payoff(node_y, cooperator_grid, L, r)
    # print(payoff_x, payoff_y)
    delta_p = payoff_x - payoff_y
    imitation_probability = 1 / (1 + np.exp(-delta_p / K))
    #      print(imitation_probability)
    if np.random.rand() < imitation_probability:
        cooperator_grid[node_x] = cooperator_grid[node_y]

    if (iteration + 1) % output_frequency == 0:
        current_cooperators = np.sum(cooperator_grid == 0) / (L * L)
        print(f"Iteration {iteration + 1}: Cooperators' ratio is {current_cooperators:.4f}")

# Final output
num_cooperators_final = np.sum(cooperator_grid == 0) / (L * L)
print(f"Final ratio of cooperators: {num_cooperators_final:.2f}")


