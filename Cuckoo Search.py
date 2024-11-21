import numpy as np
import random
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs

# Cuckoo Search Algorithm for Outlier Detection
class CuckooSearch:
    def __init__(self, num_nests, max_iter, p_a, p_s, n_features, threshold_range=(0.0, 1.0)):
        self.num_nests = num_nests
        self.max_iter = max_iter
        self.p_a = p_a  # Probability of abandoning an egg
        self.p_s = p_s  # Probability of solution
        self.n_features = n_features
        self.threshold_range = threshold_range  # Define the range of outlier threshold (0 to 1)

    def fitness(self, X, y, threshold):
        """
        Fitness function for evaluating the performance of an outlier detection method
        using a threshold. The threshold defines the outlier score.
        """
        # Calculate pairwise distances between all data points
        distances = pairwise_distances(X)
        # Find the maximum distance for each point to any other point (max distance)
        max_distances = np.max(distances, axis=1)
        # Sort distances to find outliers (larger distances are considered outliers)
        outlier_scores = max_distances / np.max(max_distances)
        
        # Count the number of outliers detected using the given threshold
        outliers = np.where(outlier_scores >= threshold)[0]
        return len(outliers)  # Fitness is the number of outliers detected

    def run(self, X, y):
        nests = np.random.uniform(self.threshold_range[0], self.threshold_range[1], self.num_nests)
        fitness_values = np.zeros(self.num_nests)
        
        # Evaluate the fitness for each nest (initial threshold)
        for i in range(self.num_nests):
            fitness_values[i] = self.fitness(X, y, nests[i])

        best_nest = np.argmax(fitness_values)  # Find the best nest based on fitness
        best_fitness = fitness_values[best_nest]
        best_threshold = nests[best_nest]
        
        # Main Cuckoo Search Loop
        for iteration in range(self.max_iter):
            for i in range(self.num_nests):
                # Abandon an egg with probability p_a and set it to the best solution
                if np.random.rand() < self.p_a:
                    nests[i] = best_threshold
                else:
                    # Otherwise, perform random walk
                    nests[i] = nests[i] + self.p_s * np.random.randn()

                # Ensure threshold is within the valid range [0, 1]
                nests[i] = np.clip(nests[i], self.threshold_range[0], self.threshold_range[1])

                # Recalculate fitness for the new threshold
                fitness_values[i] = self.fitness(X, y, nests[i])

            # Update best threshold and fitness value
            best_nest = np.argmax(fitness_values)
            best_fitness = fitness_values[best_nest]
            best_threshold = nests[best_nest]

            print(f"Iteration {iteration + 1}/{self.max_iter} - Best Threshold: {best_threshold:.3f} | Best Fitness: {best_fitness}")

        return best_threshold, best_fitness

# Main function for user input and execution
if __name__ == "__main__":
    print("Outlier Detection using Cuckoo Search Algorithm")

    # User input: Generate synthetic dataset or load user data
    choice = input("Choose dataset type (1 for synthetic, 2 for custom data): ")
    
    if choice == '1':
        # Generate a synthetic dataset with clusters
        X, y = make_blobs(n_samples=300, centers=3, random_state=42)
    else:
        # Custom user input for dataset
        num_samples = int(input("Enter number of samples: "))
        num_features = int(input("Enter number of features: "))
        
        X = np.zeros((num_samples, num_features))
        print("Enter the data values (one sample per line):")
        for i in range(num_samples):
            X[i] = np.array([float(x) for x in input(f"Sample {i + 1}: ").split()])

        y = np.zeros(num_samples)  # Labeling not needed for this unsupervised task
    
    # Set parameters for Cuckoo Search
    num_nests = 30  # Number of potential solutions (nests)
    max_iter = 100  # Maximum iterations
    p_a = 0.25  # Probability of abandoning an egg
    p_s = 0.01  # Probability of solution (random walk step size)
    n_features = X.shape[1]

    # Initialize and run Cuckoo Search for Outlier Detection
    cuckoo_search = CuckooSearch(num_nests=num_nests, max_iter=max_iter, p_a=p_a, p_s=p_s, n_features=n_features)
    best_threshold, best_fitness = cuckoo_search.run(X, y)

    # Display results
    print(f"\nBest Threshold for Outlier Detection: {best_threshold:.3f}")
    print(f"Number of Outliers Detected: {best_fitness}")
    
    # Identify and display the outliers using the best threshold
    distances = pairwise_distances(X)
    max_distances = np.max(distances, axis=1)
    outlier_scores = max_distances / np.max(max_distances)
    outliers = np.where(outlier_scores >= best_threshold)[0]
    print(f"Outliers detected at indices: {outliers}")
