import numpy as np

class GrayWolfOptimizer:
    def __init__(self, n_clusters, n_wolves, max_iter, data):
        self.n_clusters = n_clusters
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.data = data
        self.n_features = data.shape[1]

        # Initialize wolves' positions randomly (centroids)
        self.wolves = np.random.rand(self.n_wolves, self.n_clusters, self.n_features)

        # Alpha, Beta, Delta wolves
        self.alpha = None
        self.beta = None
        self.delta = None

    def fitness(self, wolf):
        # Calculate within-cluster sum of squares (WCSS) as fitness
        cluster_assignments = np.argmin(
            np.linalg.norm(self.data[:, np.newaxis, :] - wolf, axis=2), axis=1
        )
        wcss = 0
        for k in range(self.n_clusters):
            cluster_points = self.data[cluster_assignments == k]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - wolf[k]) ** 2)
        return wcss

    def update_wolves(self):
        # Sort wolves by fitness and assign alpha, beta, and delta
        fitness_values = np.array([self.fitness(wolf) for wolf in self.wolves])
        sorted_indices = np.argsort(fitness_values)
        self.alpha = self.wolves[sorted_indices[0]]
        self.beta = self.wolves[sorted_indices[1]]
        self.delta = self.wolves[sorted_indices[2]]

    def optimize(self):
        for iteration in range(self.max_iter):
            self.update_wolves()

            a = 2 - iteration * (2 / self.max_iter)  # Linear decreasing factor

            for i in range(self.n_wolves):
                for k in range(self.n_clusters):
                    r1, r2 = np.random.rand(2)
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha[k] - self.wolves[i, k])
                    X1 = self.alpha[k] - A1 * D_alpha

                    r1, r2 = np.random.rand(2)
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta[k] - self.wolves[i, k])
                    X2 = self.beta[k] - A2 * D_beta

                    r1, r2 = np.random.rand(2)
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta[k] - self.wolves[i, k])
                    X3 = self.delta[k] - A3 * D_delta

                    # Update wolf position
                    self.wolves[i, k] = (X1 + X2 + X3) / 3

        # Return the best solution (alpha wolf)
        return self.alpha

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    # Generate synthetic data
    n_samples = 300
    n_features = 2
    n_clusters = 3
    data, _ = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=42)

    # GWO parameters
    n_wolves = 10
    max_iter = 100

    gwo = GrayWolfOptimizer(n_clusters=n_clusters, n_wolves=n_wolves, max_iter=max_iter, data=data)
    best_centroids = gwo.optimize()

    print("Best centroids:", best_centroids)
