import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the Gene Expression Algorithm
class GeneExpressionAlgorithm:
    def __init__(self, population_size, generations, mutation_rate, crossover_rate, X_train, y_train):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.X_train = X_train
        self.y_train = y_train
        self.population = np.random.randint(2, size=(population_size, X_train.shape[1]))

    def fitness(self, individual):
        # Select the features based on the gene (1 means feature selected)
        selected_features = [i for i, x in enumerate(individual) if x == 1]
        
        # If no features selected, return a very low fitness score
        if len(selected_features) == 0:
            return 0
        
        # Train a classifier using only the selected features
        X_train_selected = self.X_train[:, selected_features]
        classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        classifier.fit(X_train_selected, self.y_train)
        
        # Evaluate the classifier on the training set
        y_train_pred = classifier.predict(X_train_selected)
        return accuracy_score(self.y_train, y_train_pred)
    
    def mutation(self, individual):
        # Randomly flip some bits based on mutation rate
        return [1 - bit if random.random() < self.mutation_rate else bit for bit in individual]
    
    def crossover(self, parent1, parent2):
        # Perform a one-point crossover between two parents
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        else:
            return parent1, parent2
    
    def evolve(self):
        for generation in range(self.generations):
            # Evaluate the fitness of each individual
            fitness_scores = np.array([self.fitness(individual) for individual in self.population])
            
            # Select the best individuals
            sorted_indices = np.argsort(fitness_scores)[::-1]
            self.population = self.population[sorted_indices]
            
            # Create the next generation through mutation and crossover
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = self.population[i], self.population[i + 1]
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutation(child1))
                new_population.append(self.mutation(child2))
            
            # Replace the old population with the new one
            self.population = np.array(new_population)
            
            # Print the best fitness score of this generation
            best_fitness = fitness_scores[sorted_indices[0]]
            print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")
        
        # Return the best individual after all generations
        return self.population[0]

# Run the Gene Expression Algorithm
gea = GeneExpressionAlgorithm(population_size=10, generations=20, mutation_rate=0.1, crossover_rate=0.7, X_train=X_train, y_train=y_train)
best_individual = gea.evolve()

# Evaluate the performance on the test set using the selected features
selected_features = [i for i, bit in enumerate(best_individual) if bit == 1]
X_test_selected = X_test[:, selected_features]

classifier = RandomForestClassifier(n_estimators=50, random_state=42)
classifier.fit(X_train[:, selected_features], y_train)
y_test_pred = classifier.predict(X_test_selected)

# Calculate the test accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test accuracy using selected features: {test_accuracy}")
