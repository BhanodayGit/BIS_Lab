import numpy as np
import random

# Ant Colony Optimization Parameters
num_particles = 30
max_iterations = 100
inertia_weight = 0.7
c1, c2 = 1.5, 1.5  # Cognitive and social coefficients
evaporation_rate = 0.1
alpha = 1  # Influence of pheromone
beta = 2   # Influence of heuristic information

# Power loss calculation: simple approximation based on line length and current flow
def power_loss(flow, line_length):
    return line_length * (flow**2) / 1000  # Simple power loss calculation formula

# Evaluate the fitness of the current solution (power loss)
def evaluate_power_grid(pheromones, demand, lines, capacities):
    total_loss = 0
    for i in range(len(demand)):
        for j in range(len(demand)):
            if pheromones[i][j] > 0:  # If there is a power flow between substations
                flow = pheromones[i][j]
                if flow > capacities[i][j]:
                    flow = capacities[i][j]  # Cap the flow to the maximum capacity of the line
                total_loss += power_loss(flow, lines[i][j])
    return total_loss

# Initialize pheromones
def initialize_pheromones(num_substations):
    pheromones = np.ones((num_substations, num_substations))  # Start with initial pheromone level of 1
    return pheromones

# Update pheromone levels
def update_pheromones(pheromones, particles, evaporation_rate):
    pheromones *= (1 - evaporation_rate)  # Evaporate pheromones
    for particle in particles:
        for i in range(len(particle["position"])):
            for j in range(len(particle["position"])):
                pheromones[i][j] += alpha * particle["position"][i][j]  # Add new pheromones
    return pheromones

# Update particle positions and velocities
def update_particles(particles, global_best, inertia_weight, c1, c2):
    for particle in particles:
        r1, r2 = np.random.random(), np.random.random()
        cognitive = c1 * r1 * (particle["best_position"] - particle["position"])
        social = c2 * r2 * (global_best["position"] - particle["position"])
        particle["velocity"] = inertia_weight * particle["velocity"] + cognitive + social

        # Update position and ensure it is within bounds (positive flow, not exceeding capacity)
        particle["position"] += particle["velocity"]
        particle["position"] = np.clip(particle["position"], 0, 1)  # Cap flow between 0 and 1
        particle["position"] = np.multiply(particle["position"], capacities)  # Apply capacity limits

# PSO Algorithm for Power Grid Optimization
def pso_power_grid_optimization(demand, lines, capacities):
    num_substations = len(demand)
    particles = [{"position": np.random.uniform(0, 1, (num_substations, num_substations)),
                  "velocity": np.random.uniform(-0.1, 0.1, (num_substations, num_substations)),
                  "best_position": None,
                  "best_score": float("inf")} for _ in range(num_particles)]
   
    global_best = {"position": None, "score": float("inf")}
    pheromones = initialize_pheromones(num_substations)
   
    for iteration in range(max_iterations):
        for particle in particles:
            score = evaluate_power_grid(pheromones, demand, lines, capacities)
            if score < particle["best_score"]:
                particle["best_score"] = score
                particle["best_position"] = particle["position"].copy()
           
            if score < global_best["score"]:
                global_best["score"] = score
                global_best["position"] = particle["position"].copy()

        update_pheromones(pheromones, particles, evaporation_rate)
        update_particles(particles, global_best, inertia_weight, c1, c2)
       
        print(f"Iteration {iteration + 1}/{max_iterations} - Best Power Loss: {global_best['score']:.2f}")
   
    return global_best

# Main function
if __name__ == "__main__":
    print("Power Grid Optimization using Ant Colony Optimization")
   
    num_substations = int(input("Enter the number of substations: "))
   
    print("\nEnter the power demand (MW) for each substation:")
    demand = [float(input(f"Power demand at Substation {i + 1} (MW): ")) for i in range(num_substations)]
   
    print("\nEnter the transmission line lengths (km) between substations:")
    lines = np.zeros((num_substations, num_substations))
    for i in range(num_substations):
        for j in range(i+1, num_substations):
            lines[i][j] = float(input(f"Line length between Substation {i + 1} and Substation {j + 1} (km): "))
            lines[j][i] = lines[i][j]  # Symmetric matrix for undirected lines
   
    print("\nEnter the maximum transmission line capacities (MW):")
    capacities = np.zeros((num_substations, num_substations))
    for i in range(num_substations):
        for j in range(i+1, num_substations):
            capacities[i][j] = float(input(f"Maximum capacity of line between Substation {i + 1} and Substation {j + 1} (MW): "))
            capacities[j][i] = capacities[i][j]  # Symmetric matrix for undirected lines
   
    print("\nStarting optimization process...")
    best_solution = pso_power_grid_optimization(demand, lines, capacities)

    print("\nOptimized Power Flow Distribution:")
    for i in range(num_substations):
        for j in range(i + 1, num_substations):
            print(f"Flow from Substation {i + 1} to Substation {j + 1}: {best_solution['position'][i][j]:.2f} MW")

    print(f"Total Power Loss: {best_solution['score']:.2f} MW")
