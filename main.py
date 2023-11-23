import numpy as np
import matplotlib.pyplot as plt

class AgentBasedSystem:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size))
    
    def initialize_molecules(self, num_molecules):
        indices = np.random.choice(range(self.size**2), size=num_molecules, replace=False)
        self.grid = np.zeros((self.size, self.size))
        self.grid[np.unravel_index(indices, (self.size, self.size))] = 1
    
    def diffuse(self, num_iterations):
        for _ in range(num_iterations):
            new_grid = np.zeros((self.size, self.size))
            for i in range(self.size):
                for j in range(self.size):
                    neighbors = self.get_neighbors(i, j)
                    concentration = np.sum(self.grid[neighbors]) / len(neighbors)
                    new_grid[i, j] = concentration
            self.grid = new_grid
    
    def get_neighbors(self, i, j):
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni = (i + di) % self.size
                nj = (j + dj) % self.size
                neighbors.append((ni, nj))
        return neighbors
    
    def plot(self):
        plt.imshow(self.grid, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.show()

# Example usage
size = 10
num_molecules = 20
num_iterations = 100

system = AgentBasedSystem(size)
system.initialize_molecules(num_molecules)
system.diffuse(num_iterations)
system.plot()
