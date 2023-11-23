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

                if i > 0:
                    total_concentration += self.grid[i - 1][j].concentration
                    num_adjacent_cells += 1
                if i < self.height - 1:
                    total_concentration += self.grid[i + 1][j].concentration
                    num_adjacent_cells += 1
                if j > 0:
                    total_concentration += self.grid[i][j - 1].concentration
                    num_adjacent_cells += 1
                if j < self.width - 1:
                    total_concentration += self.grid[i][j + 1].concentration
                    num_adjacent_cells += 1

                new_concentration = total_concentration / num_adjacent_cells
                new_agent = Agent(new_concentration)
                new_grid[i][j] = new_agent

        self.grid = new_grid
