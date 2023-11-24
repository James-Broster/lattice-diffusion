import os

import numpy as np
import random
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm
import tempfile
from pathlib import Path

class LatticeSimulation:
    def __init__(self, size, alpha, gamma, temp, k_B=1.0):
        self.size = size
        self.alpha = alpha
        self.gamma = gamma
        self.temp = temp
        self.k_B = k_B
        self.lattice = np.zeros(size, dtype=int)

    def initialize(self, counts):
        self.lattice = counts

    def initialize_concentrated(self, num_particles, pos):
        self.lattice[pos] += num_particles

    def initialize_random(self, num_particles):
        particle_positions = np.random.choice(self.size, num_particles)
        for pos in particle_positions:
            self.lattice[pos] += 1

    def attraction_potential(self, n):
        return -self.alpha * n**2

    def diffusion_potential(self, ni, nj):
        return self.gamma * (ni - nj)

    def total_potential(self, i):
        neighbors = [i-1, i+1]
        return self.attraction_potential(self.lattice[i]) + sum(self.diffusion_potential(self.lattice[i], self.lattice[j % self.size]) for j in neighbors)
    
    def total_potential_given_move(self, j, i):
        neighbors = [j-1, j+1]
        return self.attraction_potential(self.lattice[j]+1) + sum(self.diffusion_potential(self.lattice[j]+1, self.lattice[n % self.size]-int((n % self.size)==i)) for n in neighbors)

    def move_probability(self, i, j):
        delta_U = self.total_potential_given_move(j,i) - self.total_potential(i)
        return np.exp(-delta_U / (self.k_B * self.temp))

    def update(self):
        i = np.random.choice(self.size, 1, p=[ni/self.lattice.sum() for ni in self.lattice])
        j = i + random.choice([-1, 1])
        j = j % self.size  # Ensuring periodic boundary conditions

        if self.lattice[i] > 0 and np.random.rand() < self.move_probability(i, j):
            self.lattice[i] -= 1
            self.lattice[j] += 1

    def run_simulation(self, steps):
        for _ in tqdm(range(steps)):
            self.update()
        return self.lattice
    
    def plot_lattice(self):
        plt.figure(figsize=(10, 2))
        plt.plot(self.lattice, 'o-')
        plt.ylim(0, num_particles + 1)
        plt.title('Lattice State')
        plt.xlabel('Lattice Site')
        plt.ylabel('Number of Particles')
        plt.grid(True)


    def run_simulation_with_visualization(self, steps, gif_name='simulation.gif'):
        temp_dir = Path(tempfile.mkdtemp())  # Create a temporary directory

        filenames = []

        for step in tqdm(range(steps)):
            self.update()
            if step % 10 == 0:  # Reduce the number of frames by capturing every 10th step
                filename = temp_dir / f"step_{step}.png"  # Use the temporary directory for the filename
                self.plot_lattice()
                plt.savefig(filename)
                plt.close()
                filenames.append(filename)

        # Create an animated gif from the images
        with imageio.get_writer(gif_name, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Remove files
        for filename in filenames:
            os.remove(filename)

        # Remove the temporary directory
        os.rmdir(temp_dir)

        print(f"GIF saved as {gif_name}")


# Example usage
size = 10        # Size of the lattice
alpha = 0       # Strength of attraction
gamma = 1      # Strength of diffusion
temp = 1        # Temperature
num_particles = 50  # Total number of particles
steps = 1000      # Number of steps to simulate

# Example usage
simulation = LatticeSimulation(size, alpha, gamma, temp)
# simulation.initialize_random(num_particles=num_particles)
simulation.initialize_concentrated(num_particles, pos=5)
simulation.run_simulation_with_visualization(steps, gif_name='lattice_simulation.gif')
