import mesa
from mesa.space import MultiGrid

class ForestPatch(mesa.Agent): # The Agent, each forest patch (grid) is an "agent"
    def __init__(self, model):  #runs everytime a new patch is created
        super().__init__(model) #tells mesa to initialise agent (mesa 3.0)

        self.tree_density = 0.7     #PLACEHOLDER DATA !! (LINKED TO SENSOR DATA LATER)
        self.soil_moisture = 0.5
        self.carbon = 50
        self.state = "healthy"

    def step(self):   #runs every simulation tick, == patch's behavior.
        pass

class ForestModel(mesa.Model):     #entire forest simulation
    def __init__(self, width, height):
        super().__init__()   #proper model initialisation

        self.grid = MultiGrid(width, height, torus=False) #creates a 2D grid. torus being false ensures no wrap-around of effects
        for x in range(width):
            for y in range(height):
                patch = ForestPatch(self)
                self.grid.place_agent(patch, (x, y))

    def step(self):
        pass

if __name__ == "__main__":
    model = ForestModel(5,5)
    print("Model created successfully")
    print("Number of agents:", len(model.agents))