# E21xxx Load Flow Report Structure

## Cover Page
- Title, course, assignment, student name/ID, date

## Abstract (≤150 words)
- Objective, methods (NR implementation + PSSE comparison), key findings (voltage accuracy, losses, sensitivity ranking), conclusions

## 1. Introduction
- Problem statement and objectives
- IEEE 9-bus test system and per-unit base
- Scope: NR implementation, PSSE comparisons (NR/GS/FDLF), voltage sensitivity

## 2. Methodology
- Data: inline IEEE 9-bus parameters; per-unit base; assumptions (flat start, tol = 1e-4)
- Algorithm: Full Newton–Raphson steps (Y-bus build, mismatch, Jacobian J1–J4, linear solve, updates)
- Tools: Python script (E21xxx_LoadFlow.py), environment, libraries (numpy optional)
- PSSE runs: cases/configs, tolerances, iteration settings
- Sensitivity setup: ±10% P/Q per load bus; metrics (variance, std)

## 3. Results
### 3.1 Program Outputs (Python NR)
- Table: Bus voltages/angles (pu/deg)
- Table: Line flows and total losses; iteration count/tolerance

### 3.2 PSSE Comparisons
- Tables: PSSE NR, Gauss–Seidel, Fast Decoupled bus voltages/angles; iterations
- Table: Line flows/losses per method
- Metrics: max/mean |ΔV|, |Δδ| vs. own NR; loss differences

### 3.3 Voltage Sensitivity
- Tables/plots: Voltage profiles for –10/0/+10% per load bus
- Table: Variance/std per observed bus; ranking (max_std_any_bus, mean_std_all_buses)

## 4. Discussion
- Accuracy: deviations and causes (rounding, model options)
- Convergence: NR vs. GS vs. FDLF (speed, stability, iteration counts)
- Sensitivity insights: most influential load; implications
- Limitations: assumptions (no tap changers, etc.), numerical tolerance

## 5. Conclusion
- Key takeaways on algorithm performance and system sensitivity
- Notes on applicability and future improvements

## 6. References
- Standards/texts, PSSE manuals, cited materials

## Appendices
- A: Code listing (E21xxx_LoadFlow.py)
- B: CSV excerpts (nr_results_*.csv, sens_results_*.csv)
- C: Additional plots or full tables if trimmed in main text
