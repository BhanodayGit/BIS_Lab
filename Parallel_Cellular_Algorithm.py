import numpy as np
from multiprocessing import Pool

class ParallelCellularAlgorithm:
    def __init__(self, grid_size, max_iter):
        self.grid_size = grid_size  # Grid dimensions (rows, columns)
        self.max_iter = max_iter
        self.grid = np.zeros(grid_size, dtype=int)  # Initialize the grid

    def initialize_grid(self):
        """Initialize the traffic grid with random values representing traffic densities."""
        self.grid = np.random.randint(0, 3, size=self.grid_size)  # 0: Low, 1: Medium, 2: High traffic

    def get_neighbors(self, x, y):
        """Get the coordinates of neighboring cells."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    neighbors.append((nx, ny))
        return neighbors

    def update_cell(self, args):
        """Update traffic status of a single cell based on its neighbors."""
        x, y, grid = args
        neighbors = self.get_neighbors(x, y)

        # Rule: Traffic density in a cell is influenced by the average density of neighbors
        avg_density = np.mean([grid[nx, ny] for nx, ny in neighbors])

        if avg_density > 1.5:
            return 2  # High traffic
        elif avg_density > 0.5:
            return 1  # Medium traffic
        else:
            return 0  # Low traffic

    def update_grid(self):
        """Update the entire grid in parallel."""
        cells = [(x, y, self.grid.copy()) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])]
        with Pool() as pool:
            new_values = pool.map(self.update_cell, cells)

        # Update the grid with new values
        for i, (x, y, _) in enumerate(cells):
            self.grid[x, y] = new_values[i]

    def run(self):
        """Run the parallel cellular algorithm for traffic management."""
        self.initialize_grid()
        for iteration in range(self.max_iter):
            print(f"Iteration {iteration + 1}/{self.max_iter}")
            self.update_grid()
        return self.grid


if __name__ == "__main__":
    # Input parameters
    rows, cols = 10, 10  # Grid size
    max_iterations = 20

    # Create and run the algorithm
    pca = ParallelCellularAlgorithm(grid_size=(rows, cols), max_iter=max_iterations)
    final_grid = pca.run()

    print("Final Traffic Grid:")
    print(final_grid)
