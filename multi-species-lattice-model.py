import os

import numpy as np
import random
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm
import tempfile
from pathlib import Path
from icecream import ic

class LatticeSimulation:
    def __init__(self, size, num_species, species_sizes, alpha_matrix, gamma, temp, k_B=1.0):
        self.size = size
        self.alpha_matrix = alpha_matrix
        self.species_sizes = species_sizes
        self.gamma = gamma
        self.temp = temp
        self.k_B = k_B
        self.lattice = np.zeros((num_species, size), dtype=int)

    def initialize(self, counts):
        self.lattice = counts

    def initialize_concentrated(self, num_particles, pos):
        self.lattice[:, pos] += num_particles

    def initialize_random(self, num_particles):
        for specie in range(self.lattice.shape[0]):
            particle_positions = np.random.choice(self.size, num_particles[specie])
            for pos in particle_positions:
                self.lattice[specie, pos] += 1

    def attraction_potential(self, n_vector, species):
        return -(self.alpha_matrix @ n_vector**2)[species]

    def diffusion_potential(self, ni_vector, nj_vector):
        return self.gamma * (np.dot(ni_vector, self.species_sizes) - np.dot(nj_vector, self.species_sizes))

    def total_potential(self, i, species):
        neighbors = [i-1, i+1]
        return self.attraction_potential(self.lattice[:, i], species) + sum(self.diffusion_potential(self.lattice[:,i].squeeze(), self.lattice[:, j % self.size].squeeze()) for j in neighbors)
    
    def total_potential_given_move(self, j, i, species):
        neighbors = [j-1, j+1]
        change_vector = np.zeros(self.lattice.shape[0])
        change_vector[species] = 1
        return self.attraction_potential(self.lattice[:, j], species) + sum(self.diffusion_potential(self.lattice[:, j].squeeze()+change_vector, self.lattice[:, n % self.size].squeeze() - change_vector*int((n % self.size)==i)) for n in neighbors)

    def move_probability(self, i, j, species):
        delta_U = self.total_potential_given_move(j,i, species) - self.total_potential(i, species)
        return np.exp(-delta_U / (self.k_B * self.temp))

    def update(self):
        cell_species = np.random.choice(np.prod(self.lattice.shape), 1, p=[ni/self.lattice.sum() for ni in self.lattice.flatten()])
        species = cell_species // self.size
        i = cell_species % self.size
        j = i + random.choice([-1, 1])
        j = j % self.size  # Ensuring periodic boundary conditions

        if self.lattice[species, i].item() > 0 and np.random.rand() < self.move_probability(i, j, species):
            self.lattice[species, i] -= 1 
            self.lattice[species, j] += 1

    def run_simulation(self, steps):
        for _ in tqdm(range(steps)):
            self.update()
        return self.lattice
    
    def plot_lattice(self):
        plt.figure(figsize=(10, 2))
        for species in range(self.lattice.shape[0]):
            plt.plot(self.lattice[species], 'o-')
        plt.ylim(0, num_particles.max() + 1)
        plt.title('Lattice State')
        plt.xlabel('Lattice Site')
        plt.ylabel('Number of Particles')
        plt.grid(True)

    def run_simulation_with_visualization(self, steps, gif_name='simulation.gif'):
        temp_dir = Path(tempfile.mkdtemp())  # Create a temporary directory

        filenames = []

        for step in tqdm(range(steps)):
            self.update()
            if step % 30 == 0:  # Reduce the number of frames by capturing every 10th step
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
num_species = 3
normalised_species_sizes = np.array([3, 2, 1])
alpha = np.matrix([
    [0, 0, -0.01],
    [0, 0, -0.01],
    [0, 0, 0.01]
])              # Strength of attraction between all pairs of particles
gamma = 1      # Strength of diffusion
temp = 1        # Temperature
num_particles = np.array([100, 70, 30])
steps = 10000      # Number of steps to simulate

# Example usage
simulation = LatticeSimulation(size, num_species, normalised_species_sizes, alpha, gamma, temp)
# simulation.initialize_random(num_particles=num_particles)
simulation.initialize_random(num_particles=num_particles)
simulation.run_simulation_with_visualization(steps, gif_name='lattice_simulation.gif')
