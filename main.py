import matplotlib.pyplot as plt

# Define the system dimensions
width = 25
height = 25

# Define the initial concentrations
concentrations = [[0.0 for _ in range(width)] for _ in range(height)]
concentrations[4][4] = 100.0

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

    def diffuse(self):
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
num_timesteps = 100  # Number of diffusion steps
run.run_simulation(num_timesteps)

# Visualize the final state
run.visualize_final_state()

