import mesa
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector


class ForestPatch(mesa.Agent): # The Agent, each forest patch (grid) is an "agent"
    def __init__(self, model, tree_density, soil_moisture, carbon, state):  #runs everytime a new patch is created
        super().__init__(model) #tells mesa to initialise agent (mesa 3.0)

        self.tree_density = tree_density     #PLACEHOLDER DATA !! (LINKED TO SENSOR DATA LATER)
        self.soil_moisture = soil_moisture
        self.carbon = carbon
        self.state = state
        self.next_state = state

    def step(self):   #runs every simulation tick, == patch's behavior.
        canopy_protection = self.tree_density * 0.02 #soil drying out each tick
        drying_rate = self.model.base_drying_rate - canopy_protection

        if drying_rate < 0:
            drying_rate = 0


        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True, # uses full 8 surrounding cells around the patch
            include_center=False #dont include patch itself
        )

        stressed_neighbors = 0
        for neighbor in neighbors:
            if neighbor.state == "stressed":
                stressed_neighbors += 1
    
        stress_pressure = stressed_neighbors * 0.01 # 5 stressed neighbours = 0.05 stresspressure

        self.soil_moisture = (
            self.soil_moisture - drying_rate - stress_pressure + self.model.rainfall
        )

        if self.soil_moisture < 0:
            self.soil_moisture = 0
        if self.soil_moisture > 1:
            self.soil_moisture = 1
        if self.soil_moisture < 0.3:
            self.next_state = "stressed"
        else:
            self.next_state = "healthy"

        if self.state == "healthy":
            self.carbon += 2 * self.tree_density
        else:
            self.carbon -= 1

        if self.carbon < 0:
            self.carbon = 0


        #print(f"Density={self.tree_density:.2f}, drying={drying_rate:.3f}, moisture={self.soil_moisture:.2f}")
        #test printing keep commented

    def advance(self):
        self.state = self.next_state

class ForestModel(mesa.Model):     #entire forest simulation
    def __init__(self, width, height):
        super().__init__()   #proper model initialisation

        self.base_drying_rate = 0.03
        self.rainfall = 0.01

        self.grid = MultiGrid(width, height, torus=False) #creates a 2D grid. torus being false ensures no wrap-around of effects
        for x in range(width):
            for y in range(height):
                tree_density = self.random.uniform(0.4, 0.9)
                soil_moisture = self.random.uniform(0.2, 0.8)
                carbon = tree_density * 100
                state = "healthy"
                if soil_moisture < 0.30:
                    state = "stressed"

                patch = ForestPatch(self, tree_density, soil_moisture, carbon, state)
                self.grid.place_agent(patch, (x, y))

        self.datacollector = DataCollector(
            model_reporters={
                "TotalCarbon": ForestModel.total_carbon,
                "StressedCount": ForestModel.stressed_count,
                "AverageMoisture": ForestModel.average_moisture,
            }
        )
        self.datacollector.collect(self)

    def total_carbon(self):
        return sum(agent.carbon for agent in self.agents)

    def stressed_count(self):
        return sum(1 for agent in self.agents if agent.state == "stressed")

    def average_moisture(self):
        return sum(agent.soil_moisture for agent in self.agents) / len(self.agents)

    def step(self):
        self.agents.do("step")
        self.agents.do("advance")
        self.datacollector.collect(self)

if __name__ == "__main__":
    model = ForestModel(5, 5)
    print("Model created successfully")
    print("Number of agents:", len(model.agents))

    patches = model.agents.to_list()

    print("\nFirst 5 patches BEFORE stepping:")
    for i, patch in enumerate(patches[:5], start=1):
        print(
            f"Patch {i}: "
            f"Density={patch.tree_density:.2f}, "
            f"Moisture={patch.soil_moisture:.2f}, "
            f"Carbon={patch.carbon:.2f}, "
            f"State={patch.state}"
        )

    for i in range(5):
        model.step()
        print(f"\nAfter step {i+1}:")
        for j, patch in enumerate(patches[:5], start=1):
            print(
                f"Patch {j}: "
                f"Density={patch.tree_density:.2f}, "
                f"Moisture={patch.soil_moisture:.2f}, "
                f"Carbon={patch.carbon:.2f}, "
                f"State={patch.state}"
            )
    
    results = model.datacollector.get_model_vars_dataframe()
    print("\nModel-level results:")
    print(results)
