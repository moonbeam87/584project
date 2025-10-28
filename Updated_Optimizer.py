import pandas as pd
from z3 import *
from typing import List, Tuple

# ---------------------------
# 1. Inverter class
# ---------------------------
class Inverter:
    def __init__(self, inv_id: int, area: float, delay: float):
        self.id = inv_id
        self.area = area
        self.delay = delay  # original CSV delay

# ---------------------------
# 2. Load inverter data
# ---------------------------
def load_inverters(csv_file: str) -> List[Inverter]:
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    inverters = []
    for idx, row in df.iterrows():
        inverters.append(Inverter(
            inv_id=int(row['INV Size']),
            area=float(row['Area (um^2)']),
            delay=float(row['delay (ns)'])
        ))
    return inverters

# ---------------------------
# 3. SMT-based optimization with hard score constraint
# ---------------------------
def optimize_ring_smt_hard_score(
    inverters: List[Inverter],
    N: int,
    target_freq: float,       # GHz
    freq_tol: float,          # fraction tolerance
    max_area: float,
    a_weight: float = 0.5,
    f_weight: float = 0.5,
    score_target: float = None
):
    M = len(inverters)
    opt = Optimize()

    # Decision variables
    x = [[Bool(f"x_{i}_{j}") for j in range(M)] for i in range(N)]
    divider = Int("divider")
    div_choices = [1,2,4,8,16,32]
    opt.add(Or([divider == d for d in div_choices]))

    # One inverter per position
    for i in range(N):
        opt.add(Sum([If(x[i][j], 1, 0) for j in range(M)]) == 1)

    # Total area
    total_area = Sum([Sum([If(x[i][j], inverters[j].area, 0.0) for j in range(M)]) for i in range(N)])
    opt.add(total_area <= max_area)

    # Frequency
    total_delay = Sum([Sum([If(x[i][j], 2*inverters[j].delay, 0.0) for j in range(M)]) for i in range(N)])
    freq_actual = 1 / (4 * total_delay) / ToReal(divider)
    f_min = target_freq * (1 - freq_tol)
    f_max = target_freq * (1 + freq_tol)
    opt.add(freq_actual >= f_min)
    opt.add(freq_actual <= f_max)

    # Score
    freq_error = Abs(freq_actual - target_freq) / target_freq
    score = a_weight * total_area + f_weight * freq_error

    # Hard score constraint
    if score_target is not None:
        opt.add(score <= score_target)

    # Solve
    if opt.check() == sat:
        model = opt.model()
        chosen_inverters = []
        for i in range(N):
            for j in range(M):
                if model.evaluate(x[i][j]):
                    chosen_inverters.append(inverters[j])
                    break
        chosen_divider = model.evaluate(divider).as_long()
        freq_val = 1 / (4 * sum(2*inv.delay for inv in chosen_inverters)) / chosen_divider
        total_area_val = sum(inv.area for inv in chosen_inverters)
        freq_err_val = abs(freq_val - target_freq)/target_freq
        score_val = a_weight*total_area_val + f_weight*freq_err_val

        return {
            "N": N,
            "divider": chosen_divider,
            "frequency": freq_val,
            "total_area": total_area_val,
            "freq_error": freq_err_val,
            "score": score_val,
            "inverters": chosen_inverters
        }
    else:
        return None

# ---------------------------
# 4. Iterative minimal score search over multiple N
# ---------------------------
if __name__ == "__main__":
    inverter_files = [
        ("584-inv-VCN-0.8.csv", 0.8),
        ("584-inv-VCN-1.05.csv", 1.05),
        ("584-inv-VCN-1.3.csv", 1.3)
    ]

    N_values = [3,5,7,9,11]
    TARGET_FREQ = 0.002
    FREQ_TOL = 0.15
    MAX_AREA = 60
    A_WEIGHT = 0.5
    F_WEIGHT = 0.5

    final_best_results = []

    for csv_file, vcn in inverter_files:
        print(f"\n--- Optimizing Ring for VCN={vcn}V ({csv_file}) ---")
        inverters = load_inverters(csv_file)

        best_result = None
        for N in N_values:
            print(f"\nTrying N={N} inverters...")
            score_limit = None
            iteration = 0
            current_best = None

            while True:
                iteration += 1
                result = optimize_ring_smt_hard_score(
                    inverters, N, TARGET_FREQ, FREQ_TOL, MAX_AREA, A_WEIGHT, F_WEIGHT, score_limit
                )
                if result is None:
                    break
                else:
                    current_best = result
                    score_limit = result['score'] - 1e-6

            if current_best:
                print(f"  Best score for N={N}: {current_best['score']:.6f}")
                if (best_result is None) or (current_best['score'] < best_result['score']):
                    best_result = current_best

        if best_result:
            final_best_results.append((csv_file, vcn, best_result))
        else:
            print(f"No valid configuration found for {csv_file}.")

    # ---------------------------
    # 5. Print final optimal configurations
    # ---------------------------
    print("\n\n=== FINAL OPTIMAL CONFIGURATIONS ===")
    for csv_file, vcn, result in final_best_results:
        print(f"\nVCN={vcn}V ({csv_file}):")
        print(f"  Number of inverters: {result['N']}")
        print(f"  Divider: {result['divider']}")
        print(f"  Frequency: {result['frequency']:.6f} GHz (error {result['freq_error']*100:.2f}%)")
        print(f"  Total Area: {result['total_area']:.2f} um^2")
        print(f"  Score: {result['score']:.6f}")
        print("  Inverters used:")
        for idx, inv in enumerate(result['inverters']):
            print(f"    Pos {idx+1}: INV ID {inv.id}, Area={inv.area}, Delay={inv.delay}")
