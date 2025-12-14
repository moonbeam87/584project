# ring_optimizer_multi_modes_z3.py
from typing import List, Tuple
import itertools
from z3 import Int, Real, Solver, And, Or, sat

# ---------------------------
# 1. Inverter class
# ---------------------------
class Inverter:
    def __init__(self, inv_id: int, area: float, delay: float):
        self.id = inv_id
        self.area = area
        self.delay = delay  # ns

    def __repr__(self):
        return f"Inverter(id={self.id}, area={self.area}, delay={self.delay})"

# ---------------------------
# 2. Frequency utility
# ---------------------------
def freq_from_delays_ns(delays_ns_list: List[float], divider: int) -> float:
    total_tpd_ns = sum(delays_ns_list)
    if total_tpd_ns <= 0:
        return 0.0
    period_s = divider * 2.0 * (total_tpd_ns * 1e-9)
    freq_hz = 1.0 / period_s
    return freq_hz / 1e6  # MHz

# ---------------------------
# 3. Deterministic chain search using Z3
# ---------------------------
def deterministic_chain_search(
    inverters: List[Inverter],
    N_values: List[int],
    target_freq: float,   # MHz
    max_area: float,
    a_weight: float,
    f_weight: float,
    max_replacements: int = 2,
    freq_tol: float = 0.15
):
    M = len(inverters)
    INV_DELAYS = [inv.delay for inv in inverters]
    INV_AREAS  = [inv.area  for inv in inverters]

    div_choices = [128, 64, 32, 16, 8, 4, 2, 1]
    best_solution = None
    best_metric = None

    for divider_found in div_choices:
        for N in sorted(N_values):
            if N < 3 or N % 2 == 0:
                continue

            base_chain_idxs = [0] * N

            def candidate_accept(chain_idxs):
                area = sum(INV_AREAS[idx] for idx in chain_idxs)
                delays = [INV_DELAYS[idx] for idx in chain_idxs]
                freq = freq_from_delays_ns(delays, divider_found)

                # Z3 constraint solver
                s = Solver()
                area_var = Real('area')
                freq_var = Real('freq')
                s.add(area_var == area)
                s.add(freq_var == freq)
                s.add(area_var <= max_area)
                s.add(freq_var <= target_freq)
                s.add(freq_var >= target_freq * (1 - freq_tol))

                if s.check() != sat:
                    return None

                abs_err = abs(freq - target_freq)
                score = a_weight * area + f_weight * abs_err
                metric = (abs_err, score)
                return {
                    "chain_idxs": chain_idxs[:],
                    "freq": freq,
                    "area": area,
                    "score": score,
                    "metric": metric
                }

            # Evaluate base candidate
            cand = candidate_accept(base_chain_idxs)
            if cand and (best_solution is None or cand["metric"] < best_metric):
                best_solution = cand
                best_metric = cand["metric"]

            # Try replacements
            larger_size_indexes = list(range(1, M))
            for k in range(1, min(max_replacements, N) + 1):
                for pos_comb in itertools.combinations(range(N), k):
                    for repl_sizes in itertools.product(larger_size_indexes, repeat=k):
                        chain_idxs = base_chain_idxs[:]
                        for pos, size_idx in zip(pos_comb, repl_sizes):
                            chain_idxs[pos] = size_idx
                        candidate = candidate_accept(chain_idxs)
                        if candidate and (best_solution is None or candidate["metric"] < best_metric):
                            best_solution = candidate
                            best_metric = candidate["metric"]

        if best_solution is not None:
            break  # stop iterating dividers if a valid solution is found

    if best_solution is None:
        return None

    chain_inv_ids = [inverters[idx].id for idx in best_solution["chain_idxs"]]
    chosen_inverters = [inverters[idx] for idx in best_solution["chain_idxs"]]

    result = {
        "N": len(chain_inv_ids),
        "divider": divider_found,
        "frequency": best_solution["freq"],
        "total_area": best_solution["area"],
        "score": best_solution["score"],
        "freq_error": abs(best_solution["freq"] - target_freq) / target_freq,
        "inverter_ids": chain_inv_ids,
        "inverters": chosen_inverters
    }

    return result

# ---------------------------
# 4. Define inverter sets
# ---------------------------
# Active Mode Inverters
INVERTERS_ACTIVE = [
    Inverter(1, 3.3666, 0.24),
    Inverter(2, 3.6456, 0.18),
    Inverter(3, 3.9246, 0.145),
    Inverter(4, 4.2036, 0.124),
    Inverter(5, 4.4826, 0.109),
    Inverter(6, 4.7616, 0.098),
    Inverter(7, 5.0406, 0.089),
    Inverter(8, 5.3196, 0.081),
    Inverter(9, 5.5986, 0.075),
    Inverter(10, 5.8776, 0.069),
]

# Moderate Mode Inverters
INVERTERS_MODERATE = [
    Inverter(1, 3.3666, 3.8),
    Inverter(2, 3.6456, 3.0),
    Inverter(3, 3.9246, 2.34),
    Inverter(4, 4.2036, 1.96),
    Inverter(5, 4.4826, 1.66),
    Inverter(6, 4.7616, 1.46),
    Inverter(7, 5.0406, 1.29),
    Inverter(8, 5.3196, 1.15),
    Inverter(9, 5.5986, 1.04),
    Inverter(10, 5.8776, 0.95),
]

# Passive Mode Inverters
INVERTERS_PASSIVE = [
    Inverter(1, 3.3666, 115),
    Inverter(2, 3.6456, 89),
    Inverter(3, 3.9246, 70),
    Inverter(4, 4.2036, 59),
    Inverter(5, 4.4826, 49),
    Inverter(6, 4.7616, 42),
    Inverter(7, 5.0406, 36),
    Inverter(8, 5.3196, 32),
    Inverter(9, 5.5986, 28),
    Inverter(10, 5.8776, 25),
]


# ---------------------------
# 5. Main execution
# ---------------------------
if __name__ == "__main__":
    F_TOL = 0.1
    MODES = [
        ("ACTIVE", 4.0, INVERTERS_ACTIVE, F_TOL),
        ("MODERATE", 1.0, INVERTERS_MODERATE, F_TOL),
        ("PASSIVE", 0.25, INVERTERS_PASSIVE, F_TOL),
    ]

    N_values = [3,5,7,9,11,13,15]
    MAX_AREA = 60.0
    A_WEIGHT = 0.5
    F_WEIGHT = 0.5
    MAX_REPLACEMENTS = 2

    for mode_name, target_freq, inverter_set, freq_tol in MODES:
        print(f"\n=== Mode: {mode_name}, Target Frequency = {target_freq} MHz, Tolerance = {freq_tol*100:.1f}% ===")
        best = deterministic_chain_search(
            inverters=inverter_set,
            N_values=N_values,
            target_freq=target_freq,
            max_area=MAX_AREA,
            a_weight=A_WEIGHT,
            f_weight=F_WEIGHT,
            max_replacements=MAX_REPLACEMENTS,
            freq_tol=freq_tol
        )
        if best:
            print(f"  N = {best['N']}")
            print(f"  Divider = {best['divider']}")
            print(f"  Frequency = {best['frequency']:.6f} MHz (error {best['freq_error']*100:.4f}%)")
            print(f"  Total area = {best['total_area']:.6f} um^2")
            print(f"  Score = {best['score']:.6f}")
            print("  Inverters used (by position, INV ID, area, delay ns):")
            for i, inv in enumerate(best['inverters']):
                print(f"    Pos {i+1}: INV ID {inv.id}, Area={inv.area}, Delay={inv.delay}")
        else:
            print("No valid configuration found.")