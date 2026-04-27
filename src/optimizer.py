"""
Stage 5: Portfolio optimizer using PuLP ILP (CBC solver).
Exact binary 0/1 formulation: maximise sum of predicted fantasy points
subject to exactly 5 drivers selected and total price ≤ $100M.
Falls back to scipy.optimize.milp if PuLP/CBC unavailable.
"""
import numpy as np
import pandas as pd

BUDGET      = 100.0  # $M
TEAM_SIZE   = 5


def optimize_pulp(drivers: pd.DataFrame) -> pd.DataFrame:
    """
    drivers: DataFrame with columns [FullName, price_m, predicted_pts, ...]
    Returns the optimal 5-driver team DataFrame.
    """
    import pulp

    n      = len(drivers)
    pts    = drivers["predicted_pts"].tolist()
    prices = drivers["price_m"].tolist()
    names  = drivers["FullName"].tolist()

    prob  = pulp.LpProblem("f1_fantasy_optimizer", pulp.LpMaximize)
    x     = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    # Objective
    prob += pulp.lpSum(pts[i] * x[i] for i in range(n))

    # Budget constraint
    prob += pulp.lpSum(prices[i] * x[i] for i in range(n)) <= BUDGET

    # Exactly 5 drivers
    prob += pulp.lpSum(x[i] for i in range(n)) == TEAM_SIZE

    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"PuLP solver returned non-optimal status: {pulp.LpStatus[status]}")

    selected = [i for i in range(n) if pulp.value(x[i]) > 0.5]
    return drivers.iloc[selected].copy()


def optimize_scipy(drivers: pd.DataFrame) -> pd.DataFrame:
    """Fallback: scipy.optimize.milp."""
    from scipy.optimize import milp, LinearConstraint, Bounds

    n      = len(drivers)
    pts    = -drivers["predicted_pts"].values  # negate for minimization
    prices = drivers["price_m"].values
    ones   = np.ones(n)

    # Budget ≤ 100
    A_budget = prices.reshape(1, -1)
    # Exactly 5 drivers
    A_count  = ones.reshape(1, -1)

    A  = np.vstack([A_budget, A_count])
    lc = LinearConstraint(A, lb=[-np.inf, TEAM_SIZE], ub=[BUDGET, TEAM_SIZE])

    integrality = np.ones(n)  # all binary
    bounds = Bounds(lb=0, ub=1)

    res = milp(c=pts, constraints=lc, integrality=integrality, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"scipy.milp failed: {res.message}")

    selected = np.where(res.x > 0.5)[0]
    return drivers.iloc[selected].copy()


def optimize(drivers: pd.DataFrame) -> pd.DataFrame:
    """
    Select optimal 5-driver team. Tries PuLP first, falls back to scipy.
    drivers must have columns: FullName, price_m, predicted_pts
    """
    drivers = drivers.dropna(subset=["price_m", "predicted_pts"]).copy()
    if len(drivers) < TEAM_SIZE:
        raise ValueError(f"Need at least {TEAM_SIZE} drivers, got {len(drivers)}")

    try:
        return optimize_pulp(drivers)
    except Exception as e:
        print(f"[optimizer] PuLP failed ({e}), trying scipy fallback...")
        return optimize_scipy(drivers)


if __name__ == "__main__":
    # Quick sanity test
    import random
    random.seed(42)
    n = 20
    fake = pd.DataFrame({
        "FullName": [f"Driver {i}" for i in range(n)],
        "price_m": [round(random.uniform(10, 35), 1) for _ in range(n)],
        "predicted_pts": [round(random.uniform(0, 30), 1) for _ in range(n)],
    })
    team = optimize(fake)
    print("Optimal team:")
    print(team[["FullName", "price_m", "predicted_pts"]])
    print(f"Total price: ${team['price_m'].sum():.1f}M | Total pts: {team['predicted_pts'].sum():.1f}")
    assert len(team) == 5
    assert team["price_m"].sum() <= 100.0
    print("[optimizer] Sanity check passed ✓")
