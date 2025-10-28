import pandas as pd
from typing import List, Dict, Tuple
from enum import Enum

# --- 1. Configuration & Data Representation ---

# An Enum to define the optimization goal. This makes the code cleaner and safer.
class OptimizationGoal(Enum):
    MINIMIZE_AREA = 1       # Given a target frequency, find the smallest design.
    MAXIMIZE_FREQUENCY = 2  # Given a max area, find the fastest design.
    FIND_BEST_ERROR = 3     # Given a target frequency and area, find the most accurate design.

# The Inverter class now includes a 'vcn' attribute to track its operating voltage.
class Inverter:
    def __init__(self, inv_id: int, width: int, area: float, delay: float, vcn: float):
        self.id = inv_id
        self.width = width
        self.area = area
        self.delay = delay
        self.vcn = vcn

    def __repr__(self):
        return (f"Inverter(ID={self.id}, VCN={self.vcn}V, Area={self.area:.2f} um^2, "
                f"Delay={self.delay:.3f} ns)")

# The RingOscillator class now includes a method to calculate its score.
class RingOscillator:
    def __init__(self, inverters: List[Inverter]):
        if len(inverters) % 2 == 0:
            raise ValueError("Number of inverters (N) must be an odd integer.")
        self.inverters = inverters
        self.n = len(inverters)
        self.vcn = inverters[0].vcn if self.inverters else "N/A"

    @property
    def total_area(self) -> float:
        return sum(inv.area for inv in self.inverters)

    @property
    def frequency(self) -> float:
        total_delay_ns = sum(inv.delay for inv in self.inverters)
        period_ns = 2 * total_delay_ns
        return 1 / period_ns if period_ns > 0 else 0
        
    def get_score(self, f_target: float, a_weight: float = 0.5, f_weight: float = 0.5) -> float:
        """
        Calculates the score for this configuration based on the project report's formula.
        A lower score is better.
        Weights (a and b in the report) are simplified here as a_weight and f_weight.
        """
        freq_error = abs(f_target - self.frequency) / f_target if f_target > 0 else float('inf')
        
        # Normalize area and error to be on a similar scale. This is a simple normalization.
        # You may want to refine this based on your expected value ranges.
        normalized_area = self.total_area 
        normalized_error = freq_error * 100 # As a percentage
        
        # Weighted sum. Lower is better.
        return (a_weight * normalized_area) + (f_weight * normalized_error)

    def __repr__(self):
        inv_type = self.inverters[0].id
        return (f"RingOscillator(N={self.n}, VCN={self.vcn}V, InverterType={inv_type}, "
                f"Freq={self.frequency:.4f} GHz, Area={self.total_area:.2f} um^2)")


# --- 2. Data Loading ---
def load_inverter_data(filepaths: List[Tuple[str, float]]) -> Dict[int, Inverter]:
    """Loads inverter specifications from a list of CSV files."""
    all_inverters = {}
    key_counter = 0
    for filepath, vcn in filepaths:
        try:
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()
            for _, row in df.iterrows():
                all_inverters[key_counter] = Inverter(
                    inv_id=int(row['INV Size']), width=int(row['Gate Width (nm)']),
                    area=float(row['Area (um^2)']), delay=float(row['delay (ns)']), vcn=vcn)
                key_counter += 1
            print(f"Successfully loaded {len(df)} inverter types from '{filepath}' (VCN={vcn}V).")
        except FileNotFoundError:
            print(f"Error: The file '{filepath}' was not found.")
        except Exception as e:
            print(f"An error occurred while reading {filepath}: {e}")
    return all_inverters


# --- 3. Optimization and Search ---
def find_oscillator_configs(
    inverter_types: Dict[int, Inverter], 
    goal: OptimizationGoal,
    f_target: float, 
    max_area: float, 
    n_max: int
):
    """
    Generates and evaluates configurations based on the specified optimization goal.
    """
    print(f"\n--- Starting Search ---")
    print(f"Goal: {goal.name}")
    
    candidate_configs = []

    # Generate all possible (valid N) uniform configurations
    for n in range(3, n_max + 1, 2):
        for inv_type in inverter_types.values():
            osc = RingOscillator([inv_type] * n)
            
            # Check against the hard constraints based on the goal
            if goal == OptimizationGoal.MINIMIZE_AREA:
                # Hard constraint: Must be close to target frequency.
                if f_target > 0:
                    freq_error = abs(f_target - osc.frequency) / f_target
                    if freq_error < 0.05:
                        candidate_configs.append(osc)
            
            elif goal == OptimizationGoal.MAXIMIZE_FREQUENCY:
                # Hard constraint: Must be within the area budget.
                if osc.total_area < max_area:
                    candidate_configs.append(osc)

            elif goal == OptimizationGoal.FIND_BEST_ERROR:
                # Hard constraints: Must be within area AND close to frequency target.
                 if f_target > 0:
                     freq_error = abs(f_target - osc.frequency) / f_target
                     if freq_error < 0.05 and osc.total_area < max_area:
                         candidate_configs.append(osc)

    # --- Rank the results based on the goal ---
    if not candidate_configs:
        print("No configurations met the hard constraints.")
        return []

    if goal == OptimizationGoal.MINIMIZE_AREA:
        print(f"Finding designs with smallest area near {f_target:.2f} GHz...")
        candidate_configs.sort(key=lambda osc: osc.total_area)
        
    elif goal == OptimizationGoal.MAXIMIZE_FREQUENCY:
        print(f"Finding designs with highest frequency below {max_area:.2f} um^2...")
        candidate_configs.sort(key=lambda osc: osc.frequency, reverse=True)

    elif goal == OptimizationGoal.FIND_BEST_ERROR:
        print(f"Finding designs closest to {f_target:.2f} GHz and below {max_area:.2f} um^2...")
        candidate_configs.sort(key=lambda osc: osc.get_score(f_target))

    return candidate_configs


# --- 4. Main Execution ---
if __name__ == "__main__":
    # --- CHOOSE YOUR OPTIMIZATION GOAL ---
    #OPTIMIZATION_GOAL = OptimizationGoal.FIND_BEST_ERROR
    #OPTIMIZATION_GOAL = OptimizationGoal.MINIMIZE_AREA
    OPTIMIZATION_GOAL = OptimizationGoal.MAXIMIZE_FREQUENCY
    

    # --- Design Parameters (Constraints) ---
    TARGET_FREQ_GHZ = 0.05
    CHIP_AREA_UM2 = 300.0
    MAX_ALLOWED_AREA = 0.15 * CHIP_AREA_UM2
    MAX_INVERTERS_N = 25

    # --- Filepaths (Corrected for your project structure) ---
    inverter_files = [
        ("584-inv-VCN-0.8.csv", 0.8),
        ("584-inv-VCN-1.05.csv", 1.05),
        ("584-inv-VCN-1.3.csv", 1.3)
    ]
    
    # --- Run Search and Display Results ---
    available_inverters = load_inverter_data(inverter_files)
    
    if available_inverters:
        best_configs = find_oscillator_configs(
            available_inverters, 
            OPTIMIZATION_GOAL,
            TARGET_FREQ_GHZ, 
            MAX_ALLOWED_AREA, 
            MAX_INVERTERS_N
        )
        
        # --- MODIFIED SECTION ---
        # This now prints all valid configurations found, not just the top 5.
        print("\n--- All Valid Configurations (Ranked by Goal) ---")
        if not best_configs:
            print("No valid designs found.")
        else:
            # The loop now iterates over the entire 'best_configs' list.
            for config in best_configs:
                score = config.get_score(TARGET_FREQ_GHZ)
                freq_error = (abs(TARGET_FREQ_GHZ - config.frequency) / TARGET_FREQ_GHZ) * 100
                print(f"{config}, Freq. Error: {freq_error:.2f}%, Score: {score:.2f}")