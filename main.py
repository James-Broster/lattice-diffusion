import matplotlib.pyplot as plt
import random

# Define the system dimensions
width = 30
height = 30

# Define the initial concentrations
concentrations = [[0.0 for _ in range(width)] for _ in range(height)]
concentrations[4][4] = 600
#concentrations[35][35] = 300

# Define the AgentBasedSystem class
class Agent:
    def __init__(self, concentration):
        self.concentration = concentration

class AgentBasedSystem:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]

    def initialize_agents(self, concentrations):
        for i in range(self.height):
            for j in range(self.width):
                concentration = concentrations[i][j]
                agent = Agent(concentration)
                self.grid[i][j] = agent

    def diffuse_og(self):
        new_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                agent = self.grid[i][j]
                total_concentration = agent.concentration
                num_adjacent_cells = 1

                # Calculate adjacent cell indices with periodic boundary conditions
                left = (j - 1) % self.width
                right = (j + 1) % self.width
                up = (i - 1) % self.height
                down = (i + 1) % self.height

                total_concentration += self.grid[i][left].concentration
                total_concentration += self.grid[i][right].concentration
                total_concentration += self.grid[up][j].concentration
                total_concentration += self.grid[down][j].concentration

                num_adjacent_cells += 4

                new_concentration = total_concentration / num_adjacent_cells
                new_agent = Agent(new_concentration)
                new_grid[i][j] = new_agent

        #self.grid = new_grid

    def diffuse(self):
        new_grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                agent = self.grid[i][j]
                current_concentration = agent.concentration

                # Calculate adjacent cell indices with periodic boundary conditions
                left = (j - 1) % self.width
                right = (j + 1) % self.width
                up = (i - 1) % self.height
                down = (i + 1) % self.height

                adjacent_concentrations = [
                    self.grid[i][left].concentration,
                    self.grid[i][right].concentration,
                    self.grid[up][j].concentration,
                    self.grid[down][j].concentration
                ]

                differences = [current_concentration - concentration for concentration in adjacent_concentrations]
                probabilities = [max(0, difference) for difference in differences]
                total_probability = sum(probabilities)

                # Calculate the probabilities for staying in the current cell
                stay_probability = 1 - total_probability

                # Generate a random number between 0 and 1
                random_number = random.random()

                # Determine the new concentration based on the random number and probabilities
                if random_number < stay_probability:
                    new_concentration = current_concentration
                else:
                    cumulative_probability = 0
                    for index, probability in enumerate(probabilities):
                        cumulative_probability += probability / total_probability
                        if random_number < cumulative_probability:
                            new_concentration = adjacent_concentrations[index]
                            break

                new_agent = Agent(new_concentration)
                new_grid[i][j] = new_agent

        self.grid = new_grid

class Run:
    def __init__(self, width, height, concentrations):
        self.system = AgentBasedSystem(width, height)
        self.system.initialize_agents(concentrations)

    def run_simulation(self, num_timesteps):
        for timestep in range(num_timesteps):
            self.system.diffuse()

    def visualize_initial_state(self):
        plt.imshow(concentrations, cmap='hot', interpolation='nearest', vmin=0, vmax=2)
        plt.colorbar()
        plt.title('Initial Concentrations')
        plt.show()

    def visualize_final_state(self):
        final_concentrations = [[agent.concentration for agent in row] for row in self.system.grid]
        plt.imshow(final_concentrations, cmap='hot', interpolation='nearest', vmin=0, vmax=2)
        plt.colorbar()
        plt.title('Final Concentrations')
        plt.show()

# Create the run instance
run = Run(width, height, concentrations)

# Visualize the initial state
run.visualize_initial_state()

# Perform diffusion
num_timesteps = 80  # Number of diffusion steps
run.run_simulation(num_timesteps)

# Visualize the final state
run.visualize_final_state()

