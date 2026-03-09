import mesa
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt #visualisation/plotting
import pandas as pd # required to read CSV file into dataframe [pandas.read_csv()]

#CSV READING
def clamp(value, low, high):           #HELPER FUNCTION
    return max(low, min(high, value))  #keeps values between a min and max. clamp(value, min, max)


def load_sensor_baseline(csv_path, rows_to_average=10):
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("CSV file empty. Ensure no file corruption, or incorrect file input.")

    recent = df.tail(rows_to_average)

    return {
        "temp_c": float(recent["T_C"].mean()),
        "rh_pct": float(recent["RH_pct"].mean()),
        "soil_pct": float(recent["Soil_pct"].mean()),
        "light_d0": float(recent["LightD0"].mean()),
    }

class ForestPatch(mesa.Agent): # The Agent, each forest patch (grid) is an "agent"
    def __init__(self, model, tree_density, soil_moisture, carbon, state):  #runs everytime a new patch is created
        super().__init__(model) #tells mesa to initialise agent (mesa 3.0)

        self.tree_density = tree_density     #PLACEHOLDER DATA !! (LINKED TO SENSOR DATA LATER)
        self.soil_moisture = soil_moisture
        self.carbon = carbon
        self.state = state
        self.next_state = state
        self.fire_risk = 0.0

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
            self.soil_moisture
            - drying_rate
            - stress_pressure
            + self.model.rainfall
            + self.model.irrigation_boost
        )

        if self.soil_moisture < 0:
            self.soil_moisture = 0
        if self.soil_moisture > 1:
            self.soil_moisture = 1

        # Recalculate fire risk AFTER moisture update
        stress_ratio = stressed_neighbors / max(1, len(neighbors))
        self.fire_risk = (
            (1 - self.soil_moisture) * 0.5
            + stress_ratio * 0.3
            + (1 if self.state == "stressed" else 0) * 0.2
        )
        self.fire_risk = clamp(self.fire_risk, 0, 1)

        burning_neighbors = 0
        for neighbor in neighbors:
            if neighbor.state == "burning":
                burning_neighbors += 1

        # Carbon update based on CURRENT state
        if self.state == "healthy":
            self.carbon += 2 * self.tree_density
        elif self.state == "stressed":
            self.carbon -= 1
        elif self.state == "burning":
            self.carbon -= 5
        elif self.state == "burned":
            self.carbon -= 2

        if self.carbon < 0:
            self.carbon = 0

        # Final state decision
        if self.state == "burning":
            self.next_state = "burned"

        elif self.state == "burned":
            self.next_state = "burned"

        elif self.fire_risk >= self.model.ignition_threshold:
            self.next_state = "burning"

        elif burning_neighbors > 0 and self.fire_risk >= self.model.spread_threshold:
            self.next_state = "burning"

        elif self.soil_moisture < 0.3:
            self.next_state = "stressed"

        else:
            self.next_state = "healthy"


        #print(f"Density={self.tree_density:.2f}, drying={drying_rate:.3f}, moisture={self.soil_moisture:.2f}")
        #test printing keep commented

    def advance(self):
        self.state = self.next_state

class ForestModel(mesa.Model):     #entire forest simulation
    def __init__(
            self,
            width,
            height,
            csv_path=None,
            base_drying_rate=None,
            rainfall=None,
            density_min=0.4,
            density_max=0.9,
            adaptive_enabled=True,
            ignition_threshold=0.75,
            spread_threshold=0.55,
            seed=None
    ):
        super().__init__(seed=seed)   #proper model initialisation

        if csv_path is not None:
            sensor = load_sensor_baseline(csv_path)

            self.base_soil_moisture = clamp(sensor["soil_pct"] / 100.0, 0, 1)
            self.base_temp_c = sensor["temp_c"]
            self.base_rh = clamp(sensor["rh_pct"] / 100.0, 0, 1)
        else:  #fallback to default values if fails
            self.base_soil_moisture = 0.5
            self.base_temp_c = 20.0
            self.base_rh = 0.6

        self.ignition_threshold = ignition_threshold  #own model design, allows me to make wildfire escalation
        self.spread_threshold = spread_threshold # a proper scenario by changing thresholds.

        calculated_drying_rate = (0.02 + max(0, self.base_temp_c - 20) * 0.0015 + (1- self.base_rh) * 0.01)

        if base_drying_rate is None:
            self.base_drying_rate = calculated_drying_rate
        else:
            self.base_drying_rate = base_drying_rate

        if rainfall is None:
            self.rainfall = 0.01
        else:
            self.rainfall = rainfall

        self.density_min = density_min
        self.density_max = density_max

        self.adaptive_enabled = adaptive_enabled #ADAPTIVE SYSTEM TOGGLE (in ForestModel(x,y, adaptive_enabled=True/False))
        self.alert = False 
        self.irrigation_boost = 0.0 #Dynamically changing Irrigation Boost
        self.stress_threshold = 0.25 #if more than 25% of patches are stressed, enable adaptive system

        self.grid = MultiGrid(width, height, torus=False) #creates a 2D grid. torus being false ensures no wrap-around of effects
        for x in range(width):
            for y in range(height):
                tree_density = self.random.uniform(self.density_min, self.density_max)
                soil_moisture = clamp(
                    self.base_soil_moisture + self.random.uniform(-0.10, 0.10),0,1
                )   #patch BEGINS around sensor-measured soil moisture, but slightly varies patch-to-patch for realism.
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
                "Alert": lambda m: int(m.alert), #stores alert on/off as number 1 or 0
                "IrrigationBoost": lambda m: m.irrigation_boost,
                "AverageFireRisk": ForestModel.average_fire_risk,
                "BurningCount": ForestModel.burning_count,
                "BurnedCount": ForestModel.burned_count,
            }
        )
        self.datacollector.collect(self)

    def total_carbon(self):
        return sum(agent.carbon for agent in self.agents)

    def stressed_count(self):
        return sum(1 for agent in self.agents if agent.state == "stressed")

    def average_moisture(self):
        return sum(agent.soil_moisture for agent in self.agents) / len(self.agents)
    
    def average_fire_risk(self):
        return sum(agent.fire_risk for agent in self.agents) /len(self.agents)
    
    def burning_count(self):
        return sum(1 for agent in self.agents if agent.state == "burning")
    
    def burned_count(self):
        return sum(1 for agent in self.agents if agent.state == "burned")
    
    def update_adaptive_response(self):
        stressed_ratio = self.stressed_count() / len(self.agents)

        if self.adaptive_enabled and stressed_ratio >= self.stress_threshold:
            self.alert = True
            self.irrigation_boost = 0.02
        else:
            self.alert = False
            self.irrigation_boost = 0.0

    def step(self):
        self.update_adaptive_response() #check overall forest stress
        self.agents.do("step") #decision
        self.agents.do("advance") #patch update
        self.datacollector.collect(self) #collect results

def run_scenario(name, steps, width=20, height=20, **model_kwargs):
    model = ForestModel(width,height, **model_kwargs)

    for _ in range(steps):
        model.step()

    results = model.datacollector.get_model_vars_dataframe()
    results["Scenario"] = name
    return results

if __name__ == "__main__":
    model = ForestModel(20, 20, csv_path="sensor_data.csv", seed=42)
    print("Model created successfully")
    print("Number of agents:", len(model.agents))

    patches = model.agents.to_list()

    print("Base soil moisture:", f"{model.base_soil_moisture:.2f}")
    print("Base temperature:", model.base_temp_c)
    print("Base humidity:", f"{model.base_rh:.2f}")
    print("Base drying rate:", f"{model.base_drying_rate:.4f}")

    print("\nFirst 5 patches BEFORE stepping:")
    for i, patch in enumerate(patches[:5], start=1):
        print(
            f"Patch {i}: "
            f"Density={patch.tree_density:.2f}, "
            f"Moisture={patch.soil_moisture:.2f}, "
            f"Carbon={patch.carbon:.2f}, "
            f"State={patch.state}"
        )

    for i in range(30):
        model.step()
        # print(f"\nAfter step {i+1}:")
        # for j, patch in enumerate(patches[:5], start=1):
        #     print(
        #         f"Patch {j}: "
        #         f"Density={patch.tree_density:.2f}, "
        #         f"Moisture={patch.soil_moisture:.2f}, "
        #         f"Carbon={patch.carbon:.2f}, "
        #         f"State={patch.state}"
        #     )
    
    baseline = run_scenario(
        "Baseline",
        30,
        csv_path="sensor_data.csv",
        base_drying_rate=0.03,
        rainfall=0.01,
        density_min=0.4,
        density_max=0.9,
        seed=42
    )

    drought = run_scenario(
        "Drought",
        30,
        csv_path="sensor_data.csv",
        base_drying_rate=0.05,
        rainfall=0.003,
        density_min=0.4,
        density_max=0.9,
        seed=42
    )

    wildfire = run_scenario( #lower canopy protection, easier ignition, easier spread scenario
        "Wildfire Escalation",
        30,
        csv_path="sensor_data.csv",
        base_drying_rate=0.04,
        rainfall=0.006,
        density_min=0.25,
        density_max=0.65,
        ignition_threshold=0.60,
        spread_threshold=0.45,
        seed=42
    )  

    afforestation = run_scenario(
        "Afforestation",
        30,
        csv_path="sensor_data.csv",
        base_drying_rate=0.03,
        rainfall=0.01,
        density_min=0.65,
        density_max=1.0,
        seed=42
    )

    no_adapt = run_scenario(
        "No Adaptation",
        30,
        csv_path="sensor_data.csv",
        base_drying_rate=0.03,
        rainfall=0.01,
        density_min=0.4,
        density_max=0.9,
        adaptive_enabled=False,
        seed=42
    )

    with_adapt = run_scenario(
        "With Adaptation",
        30,
        csv_path="sensor_data.csv",
        base_drying_rate=0.03,
        rainfall=0.01,
        density_min=0.4,
        density_max=0.9,
        adaptive_enabled=True,
        seed=42
    )

    results = model.datacollector.get_model_vars_dataframe()
    print("\nModel-level results:")
    print(results)

    plt.figure()
    plt.plot(results["TotalCarbon"])
    plt.xlabel("Step")
    plt.ylabel("Total Carbon")
    plt.title("Total Carbon Over Time")
    plt.show()

    plt.figure()
    plt.plot(results["StressedCount"])
    plt.xlabel("Step")
    plt.ylabel("Stressed Patches")
    plt.title("Stressed Patches Over Time")
    plt.show()

    plt.figure()
    plt.plot(results["AverageMoisture"])
    plt.xlabel("Step")
    plt.ylabel("Average Moisture")
    plt.title("Average Moisture Over Time")
    plt.show()

    plt.figure()
    plt.plot(baseline["TotalCarbon"], label="Baseline")
    plt.plot(drought["TotalCarbon"], label="Drought")
    plt.plot(afforestation["TotalCarbon"], label="Wildfire Escalation")
    plt.xlabel("Step")
    plt.ylabel("Total Carbon")
    plt.title("Scenario Comparison: Total Carbon")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(results["Alert"])
    plt.xlabel("Step")
    plt.ylabel("Alert State")
    plt.title("Adaptive Alert Over Time")
    plt.show()

    plt.figure()
    plt.plot(results["IrrigationBoost"])
    plt.xlabel("Step")
    plt.ylabel("Irrigation Boost")
    plt.title("Adaptive Moisture Support Over Time")
    plt.show()

    plt.figure()
    plt.plot(no_adapt["TotalCarbon"], label="No Adaptation")
    plt.plot(with_adapt["TotalCarbon"], label="With Adaptation")
    plt.xlabel("Step")
    plt.ylabel("Total Carbon")
    plt.title("Adaptive Comparison: Total Carbon")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(no_adapt["StressedCount"], label="No Adaptation")
    plt.plot(with_adapt["StressedCount"], label="With Adaptation")
    plt.xlabel("Step")
    plt.ylabel("Stressed Patches")
    plt.title("Adaptive Comparison: Stressed Patches")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(no_adapt["AverageMoisture"], label="No Adaptation")
    plt.plot(with_adapt["AverageMoisture"], label="With Adaptation")
    plt.xlabel("Step")
    plt.ylabel("Average Moisture")
    plt.title("Adaptive Comparison: Average Moisture")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(with_adapt["Alert"], label="With Adaptation")
    plt.xlabel("Step")
    plt.ylabel("Alert State")
    plt.title("Adaptive Alert Activation")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(results["AverageFireRisk"])
    plt.xlabel("Step")
    plt.ylabel("Average Fire Risk")
    plt.title("Average Fire Risk Over Time")
    plt.show()

    plt.figure()
    plt.plot(results["BurningCount"])
    plt.xlabel("Step")
    plt.ylabel("Burning Patches")
    plt.title("Burning Patches Over Time")
    plt.show()

    plt.figure()
    plt.plot(results["BurnedCount"])
    plt.xlabel("Step")
    plt.ylabel("Burned Patches")
    plt.title("Burned Patches Over Time")
    plt.show()

    plt.figure()
    plt.plot(drought["AverageFireRisk"], label="Drought")
    plt.plot(wildfire["AverageFireRisk"], label="Wildfire Escalation")
    plt.xlabel("Step")
    plt.ylabel("Average Fire Risk")
    plt.title("Scenario Comparison: Fire Risk")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(drought["BurnedCount"], label="Drought")
    plt.plot(wildfire["BurnedCount"], label="Wildfire Escalation")
    plt.xlabel("Step")
    plt.ylabel("Burned Patches")
    plt.title("Scenario Comparison: Burned Patches")
    plt.legend()
    plt.show()