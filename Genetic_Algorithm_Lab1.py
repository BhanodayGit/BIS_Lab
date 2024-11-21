import numpy as np
import random

# Problem Definition: Items with weights and values
items = [
    {"weight": 2, "value": 3},
    {"weight": 3, "value": 4},
    {"weight": 4, "value": 5},
    {"weight": 5, "value": 8},
    {"weight": 9, "value": 10}
]
capacity = 15  # Knapsack capacity

# Genetic Algorithm Parameters
POPULATION_SIZE = 10
GENERATIONS = 50
MUTATION_RATE = 0.1

# Fitness Function: Calculate total value and enforce weight constraint
def fitness_function(individual):
    total_weight = total_value = 0
    for i, gene in enumerate(individual):
        if gene == 1:
            total_weight += items[i]["weight"]
            total_value += items[i]["value"]
    # Penalize solutions exceeding capacity
    if total_weight > capacity:
        return 0
    return total_value

# Create the initial population
def initialize_population():
    return [np.random.randint(0, 2, len(items)).tolist() for _ in range(POPULATION_SIZE)]

# Perform crossover between two parents
def crossover(parent1, parent2):
    point = random.randint(1, len(items) - 1)
    return parent1[:point] + parent2[point:]

# Apply mutation to an individual
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]  # Flip bit (0 -> 1, 1 -> 0)
    return individual

# Select parents using tournament selection
def select_parents(population, fitness_scores):
    tournament = random.sample(range(len(population)), 3)
    best_index = max(tournament, key=lambda idx: fitness_scores[idx])
    return population[best_index]

# Genetic Algorithm
def genetic_algorithm():
    population = initialize_population()
    best_solution = None
    best_fitness = 0

    for generation in range(GENERATIONS):
        fitness_scores = [fitness_function(ind) for ind in population]
        new_population = []

        # Save the best solution
        max_fitness_idx = np.argmax(fitness_scores)
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_solution = population[max_fitness_idx]

        # Generate new population
        for _ in range(POPULATION_SIZE // 2):
            parent1 = select_parents(population, fitness_scores)
            parent2 = select_parents(population, fitness_scores)
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            new_population.extend([child1, child2])

        population = new_population
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    return best_solution, best_fitness

# Run the Genetic Algorithm
solution, max_value = genetic_algorithm()
print("\nOptimal Solution:", solution)
print("Maximum Value Achieved:", max_value)

# Display the selected items
selected_items = [i for i, gene in enumerate(solution) if gene == 1]
print("Selected Items:", selected_items)
for i in selected_items:
    print(f"Item {i + 1}: Weight = {items[i]['weight']}, Value = {items[i]['value']}")
