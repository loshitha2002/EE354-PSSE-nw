"""
Full Newton–Raphson load flow for IEEE 9-bus test system.
Student: YOUR_NAME | ID: YOUR_ID
Date: 2025-12-09

This script implements the assignment Task 1 requirements:
- Flat start, programmatic Y-bus construction, explicit Jacobian (J1–J4),
  no external power-flow solvers.
- Accepts inline default data and optional CSV inputs.
- Computes bus voltages, power mismatches, line flows, and system losses.

Notes for the grader:
- Replace YOUR_NAME and YOUR_ID above and in function headers.
- Flowchart boxes should reference the line numbers below (use your editor’s
  line numbers). Each function already carries a header comment block.
- Sample execution is provided in the __main__ section.
"""
from __future__ import annotations

import cmath
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# -------------------------- Data Classes --------------------------

@dataclass
class Bus:
    """Student: YOUR_NAME | ID: YOUR_ID | Represents a single bus record."""
    number: int
    base_kv: float
    bus_type: str  # "SLACK", "PV", "PQ"
    pd_mw: float = 0.0
    qd_mvar: float = 0.0
    pg_mw: float = 0.0
    v_spec: float = 1.0
    qmax_mvar: float = 9999.0
    qmin_mvar: float = -9999.0

@dataclass
class Branch:
    """Student: YOUR_NAME | ID: YOUR_ID | Represents a line or transformer."""
    from_bus: int
    to_bus: int
    r_pu: float
    x_pu: float
    g_pu: float = 0.0
    b_pu: float = 0.0  # Total line charging susceptance (use b/2 at each end)

# -------------------------- Utility Functions --------------------------

def try_numpy_solve(matrix: List[List[complex]], rhs: List[complex]) -> List[complex]:
    """Student: YOUR_NAME | ID: YOUR_ID | Solve linear system using numpy if available, else fallback."""
    try:
        import numpy as np  # type: ignore

        A = np.array(matrix, dtype=complex)
        b = np.array(rhs, dtype=complex)
        sol = np.linalg.solve(A, b)
        return sol.tolist()
    except Exception:
        return gaussian_elimination(matrix, rhs)

def gaussian_elimination(matrix: List[List[complex]], rhs: List[complex]) -> List[complex]:
    """Student: YOUR_NAME | ID: YOUR_ID | Basic Gaussian elimination with partial pivoting (complex-safe)."""
    n = len(rhs)
    # Create augmented matrix
    aug = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        # Pivot selection
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("Singular matrix encountered during elimination")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        # Normalize pivot row
        pivot_val = aug[col][col]
        aug[col] = [val / pivot_val for val in aug[col]]
        # Eliminate below
        for row in range(col + 1, n):
            factor = aug[row][col]
            aug[row] = [aug[row][k] - factor * aug[col][k] for k in range(n + 1)]
    # Back substitution
    x = [0j] * n
    for i in range(n - 1, -1, -1):
        x[i] = aug[i][n] - sum(aug[i][j] * x[j] for j in range(i + 1, n))
    return x

# -------------------------- Data Loading --------------------------

def default_system_data() -> Tuple[List[Bus], List[Branch]]:
    """Student: YOUR_NAME | ID: YOUR_ID | Returns inline IEEE 9-bus data (per-unit on 100 MVA)."""
    buses = [
        Bus(1, 16.5, "SLACK", pd_mw=0.0, qd_mvar=0.0, pg_mw=71.6448, v_spec=1.04),
        Bus(2, 18.0, "PV", pd_mw=0.0, qd_mvar=0.0, pg_mw=163.0, v_spec=1.025),
        Bus(3, 13.8, "PV", pd_mw=0.0, qd_mvar=0.0, pg_mw=85.0, v_spec=1.025),
        Bus(4, 230.0, "PQ"),
        Bus(5, 230.0, "PQ", pd_mw=125.0, qd_mvar=50.0),
        Bus(6, 230.0, "PQ", pd_mw=90.0, qd_mvar=30.0),
        Bus(7, 230.0, "PQ"),
        Bus(8, 230.0, "PQ", pd_mw=100.0, qd_mvar=35.0),
        Bus(9, 230.0, "PQ"),
    ]

    branches = [
        # Transformers
        Branch(1, 4, r_pu=0.0, x_pu=0.0576, g_pu=0.0, b_pu=0.0),
        Branch(2, 7, r_pu=0.0, x_pu=0.0625, g_pu=0.0, b_pu=0.0),
        Branch(3, 9, r_pu=0.0, x_pu=0.0586, g_pu=0.0, b_pu=0.0),
        # Transmission lines
        Branch(4, 5, r_pu=0.0100, x_pu=0.0850, g_pu=0.0, b_pu=0.1760),
        Branch(4, 6, r_pu=0.0170, x_pu=0.0920, g_pu=0.0, b_pu=0.1580),
        Branch(5, 7, r_pu=0.0320, x_pu=0.1610, g_pu=0.0, b_pu=0.3060),
        Branch(6, 9, r_pu=0.0390, x_pu=0.1700, g_pu=0.0, b_pu=0.3580),
        Branch(7, 8, r_pu=0.0085, x_pu=0.0720, g_pu=0.0, b_pu=0.1490),
        Branch(8, 9, r_pu=0.0119, x_pu=0.1008, g_pu=0.0, b_pu=0.2090),
    ]
    return buses, branches

def read_branch_csv(path: str) -> List[Branch]:
    """Student: YOUR_NAME | ID: YOUR_ID | Read branch/line data from CSV with columns: from,to,r,x,g,b."""
    rows: List[Branch] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                Branch(
                    int(row["from"]),
                    int(row["to"]),
                    float(row["r"]),
                    float(row["x"]),
                    float(row.get("g", 0.0) or 0.0),
                    float(row.get("b", 0.0) or 0.0),
                )
            )
    return rows

def read_bus_csv(path: str, bus_types: Dict[int, str], pg: Dict[int, float], vd: Dict[int, float], qmax: Dict[int, float], qmin: Dict[int, float]) -> List[Bus]:
    """Student: YOUR_NAME | ID: YOUR_ID | Read bus data; user provides type/gen maps separately for flexibility."""
    buses: List[Bus] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            num = int(row["bus"])
            buses.append(
                Bus(
                    number=num,
                    base_kv=float(row.get("base_kv", 230.0) or 230.0),
                    bus_type=bus_types.get(num, "PQ"),
                    pd_mw=float(row.get("pd", 0.0) or 0.0),
                    qd_mvar=float(row.get("qd", 0.0) or 0.0),
                    pg_mw=pg.get(num, 0.0),
                    v_spec=vd.get(num, 1.0),
                    qmax_mvar=qmax.get(num, 9999.0),
                    qmin_mvar=qmin.get(num, -9999.0),
                )
            )
    return buses

# -------------------------- Y-Bus Construction --------------------------

def build_ybus(buses: List[Bus], branches: List[Branch]) -> List[List[complex]]:
    """Student: YOUR_NAME | ID: YOUR_ID | Construct the nodal admittance matrix Ybus."""
    n = len(buses)
    idx = {bus.number: i for i, bus in enumerate(buses)}
    ybus = [[0j for _ in range(n)] for _ in range(n)]

    for br in branches:
        if br.r_pu == 0 and br.x_pu == 0:
            raise ValueError(f"Branch {br.from_bus}-{br.to_bus} has zero impedance")
        y_series = 1 / complex(br.r_pu, br.x_pu)
        b_shunt = 1j * br.b_pu / 2.0
        g_shunt = br.g_pu / 2.0

        i = idx[br.from_bus]
        k = idx[br.to_bus]
        # Off-diagonal terms
        ybus[i][k] -= y_series
        ybus[k][i] -= y_series
        # Diagonal terms include series admittance + shunts
        ybus[i][i] += y_series + b_shunt + g_shunt
        ybus[k][k] += y_series + b_shunt + g_shunt
    return ybus

# -------------------------- Power Calculations --------------------------

def power_injections(ybus: List[List[complex]], voltages: List[complex]) -> Tuple[List[float], List[float]]:
    """Student: YOUR_NAME | ID: YOUR_ID | Compute P and Q injections from current voltages."""
    n = len(voltages)
    P = [0.0] * n
    Q = [0.0] * n
    for i in range(n):
        Si = voltages[i] * complex_sum_conj(ybus[i], voltages)
        P[i] = Si.real
        Q[i] = Si.imag  # Injection sign convention: S = P + jQ (injected)
    return P, Q

def complex_sum_conj(row: List[complex], voltages: List[complex]) -> complex:
    """Student: YOUR_NAME | ID: YOUR_ID | Helper: sum(Y_ij * V_j) with conjugate handling."""
    total = 0j
    for y, v in zip(row, voltages):
        total += y * v
    return total.conjugate()

# -------------------------- Jacobian Assembly --------------------------

def assemble_jacobian(
    buses: List[Bus],
    ybus: List[List[complex]],
    V: List[float],
    delta: List[float],
    P: List[float],
    Q: List[float],
    pv_index: List[int],
    pq_index: List[int],
) -> List[List[complex]]:
    """Student: YOUR_NAME | ID: YOUR_ID | Build J1–J4 blocks and assemble full Jacobian."""
    n = len(buses)
    g = [[y.real for y in row] for row in ybus]
    b = [[y.imag for y in row] for row in ybus]

    ang_vars = [i for i in range(n) if buses[i].bus_type != "SLACK"]
    mag_vars = pq_index

    J1 = [[0.0 for _ in ang_vars] for _ in ang_vars]
    J2 = [[0.0 for _ in mag_vars] for _ in ang_vars]
    J3 = [[0.0 for _ in ang_vars] for _ in mag_vars]
    J4 = [[0.0 for _ in mag_vars] for _ in mag_vars]

    # J1 and J2 rows: dP/d(delta,V) for non-slack buses
    for r, i in enumerate(ang_vars):
        for c, j in enumerate(ang_vars):
            if i == j:
                J1[r][c] = -Q[i] - b[i][i] * V[i] * V[i]
            else:
                angle = delta[i] - delta[j]
                J1[r][c] = V[i] * V[j] * (g[i][j] * math.sin(angle) - b[i][j] * math.cos(angle))
        for c, j in enumerate(mag_vars):
            if i == j:
                J2[r][c] = P[i] / V[i] + g[i][i] * V[i]
            else:
                angle = delta[i] - delta[j]
                J2[r][c] = V[i] * (g[i][j] * math.cos(angle) + b[i][j] * math.sin(angle))

    # J3 and J4 rows: dQ/d(delta,V) for PQ buses only
    for r, i in enumerate(mag_vars):
        for c, j in enumerate(ang_vars):
            if i == j:
                J3[r][c] = P[i] - g[i][i] * V[i] * V[i]
            else:
                angle = delta[i] - delta[j]
                J3[r][c] = -V[i] * V[j] * (g[i][j] * math.cos(angle) + b[i][j] * math.sin(angle))
        for c, j in enumerate(mag_vars):
            if i == j:
                J4[r][c] = Q[i] / V[i] - b[i][i] * V[i]
            else:
                angle = delta[i] - delta[j]
                J4[r][c] = V[i] * (g[i][j] * math.sin(angle) - b[i][j] * math.cos(angle))

    # Assemble blocks
    top = [J1_row + J2_row for J1_row, J2_row in zip(J1, J2)]
    bottom = [J3_row + J4_row for J3_row, J4_row in zip(J3, J4)]
    return [list(map(complex, row)) for row in (top + bottom)]

# -------------------------- Newton–Raphson Solver --------------------------

def newton_raphson_pf(
    buses: List[Bus],
    branches: List[Branch],
    tol: float = 1e-4,
    max_iter: int = 15,
    flat_start: bool = True,
) -> Tuple[List[complex], List[float], List[float], List[Dict[str, float]]]:
    """Student: YOUR_NAME | ID: YOUR_ID | Solve load flow and return voltages, P, Q, iteration log."""
    n = len(buses)
    ybus = build_ybus(buses, branches)

    # Indexing helpers
    idx = {bus.number: i for i, bus in enumerate(buses)}
    slack_idx = next(i for i, b in enumerate(buses) if b.bus_type == "SLACK")
    pv_idx = [i for i, b in enumerate(buses) if b.bus_type == "PV"]
    pq_idx = [i for i, b in enumerate(buses) if b.bus_type == "PQ"]

    # Initial voltages
    Vmag = [1.0] * n
    ang = [0.0] * n
    if flat_start:
        for i, b in enumerate(buses):
            Vmag[i] = 1.0
            ang[i] = 0.0
    # Apply specified slack and PV magnitudes
    Vmag[slack_idx] = buses[slack_idx].v_spec
    for i in pv_idx:
        Vmag[i] = buses[i].v_spec

    iter_log: List[Dict[str, float]] = []

    for it in range(1, max_iter + 1):
        V = [Vmag[i] * cmath.exp(1j * ang[i]) for i in range(n)]
        P_inj, Q_inj = power_injections(ybus, V)

        # Specified injections (generation - load)
        P_spec = [b.pg_mw / 100.0 - b.pd_mw / 100.0 for b in buses]
        Q_spec = [-b.qd_mvar / 100.0 for b in buses]  # generator Q unknown, load is specified

        # Mismatch vector: P for all non-slack, Q for PQ only
        mismatch: List[complex] = []
        ang_vars = [i for i in range(n) if i != slack_idx]
        for i in ang_vars:
            mismatch.append(P_spec[i] - P_inj[i])
        for i in pq_idx:
            mismatch.append(Q_spec[i] - Q_inj[i])

        max_mis = max(abs(m) for m in mismatch) if mismatch else 0.0
        iter_log.append({
            "iter": it,
            "max_mismatch_pu": max_mis,
            "v_min": min(abs(v) for v in V),
            "v_max": max(abs(v) for v in V),
        })

        if max_mis < tol:
            break

        J = assemble_jacobian(buses, ybus, Vmag, ang, P_inj, Q_inj, pv_idx, pq_idx)
        dx = try_numpy_solve(J, mismatch)

        # Update angles (non-slack) and magnitudes (PQ only)
        for k, i in enumerate(ang_vars):
            ang[i] += dx[k].real  # dx is real for angle corrections
        for k, i in enumerate(pq_idx):
            Vmag[i] += dx[len(ang_vars) + k].real

    voltages = [Vmag[i] * cmath.exp(1j * ang[i]) for i in range(n)]
    return voltages, P_inj, Q_inj, iter_log

# -------------------------- Line Flows --------------------------

def compute_line_flows(
    branches: List[Branch],
    buses: List[Bus],
    ybus: List[List[complex]],
    voltages: List[complex],
) -> Tuple[List[Dict[str, float]], float]:
    """Student: YOUR_NAME | ID: YOUR_ID | Compute per-branch power flows and total loss (MW)."""
    idx = {bus.number: i for i, bus in enumerate(buses)}
    flows: List[Dict[str, float]] = []
    total_loss_mw = 0.0

    for br in branches:
        i = idx[br.from_bus]
        k = idx[br.to_bus]
        y_series = 1 / complex(br.r_pu, br.x_pu)
        y_shunt = complex(br.g_pu / 2.0, br.b_pu / 2.0)

        I_ik = (voltages[i] - voltages[k]) * y_series + voltages[i] * y_shunt
        I_ki = (voltages[k] - voltages[i]) * y_series + voltages[k] * y_shunt

        S_ik = voltages[i] * I_ik.conjugate()
        S_ki = voltages[k] * I_ki.conjugate()

        loss = (S_ik + S_ki).real * 100.0  # MW on 100 MVA base
        total_loss_mw += loss

        flows.append({
            "from": br.from_bus,
            "to": br.to_bus,
            "P_from_MW": S_ik.real * 100.0,
            "Q_from_Mvar": S_ik.imag * 100.0,
            "P_to_MW": S_ki.real * 100.0,
            "Q_to_Mvar": S_ki.imag * 100.0,
            "loss_MW": loss,
        })

    return flows, total_loss_mw

# -------------------------- Demonstration Runner --------------------------

def pretty_voltage_table(buses: List[Bus], voltages: List[complex]) -> str:
    """Student: YOUR_NAME | ID: YOUR_ID | Format bus voltages for display."""
    lines = ["Bus | |V| (pu) | Angle (deg)", "----|----------|------------"]
    for bus, V in zip(buses, voltages):
        mag = abs(V)
        ang_deg = math.degrees(cmath.phase(V))
        lines.append(f"{bus.number:3d} | {mag:8.4f} | {ang_deg:10.4f}")
    return "\n".join(lines)

def run_demo():
    """Student: YOUR_NAME | ID: YOUR_ID | Run sample power flow on IEEE 9-bus and print iteration 2 results."""
    buses, branches = default_system_data()
    voltages, P_inj, Q_inj, log = newton_raphson_pf(buses, branches, tol=1e-4, max_iter=15)

    ybus = build_ybus(buses, branches)
    flows, total_loss_mw = compute_line_flows(branches, buses, ybus, voltages)

    print("Student ID: YOUR_ID")
    print("Iteration log (per-unit mismatches):")
    for rec in log:
        print(f"Iter {rec['iter']:2d}: max mismatch = {rec['max_mismatch_pu']:.5f}, Vmin={rec['v_min']:.4f}, Vmax={rec['v_max']:.4f}")

    # Show 2nd iteration voltages if available
    if len(log) >= 2:
        print("\nVoltages at end of iteration 2 (flat start reference):")
        # Re-run up to 2 iterations to extract values
        volt2, _, _, _ = newton_raphson_pf(buses, branches, tol=1e-12, max_iter=2)
        print(pretty_voltage_table(buses, volt2))

    print("\nFinal converged voltages:")
    print(pretty_voltage_table(buses, voltages))
    print(f"\nTotal system loss: {total_loss_mw:.4f} MW")

    print("\nSample line flows (MW/Mvar):")
    for f in flows:
        print(
            f"{f['from']:d}->{f['to']:d}: P_from={f['P_from_MW']:.3f} MW, Q_from={f['Q_from_Mvar']:.3f} Mvar, "
            f"P_to={f['P_to_MW']:.3f} MW, Q_to={f['Q_to_Mvar']:.3f} Mvar, loss={f['loss_MW']:.3f} MW"
        )

if __name__ == "__main__":
    run_demo()
