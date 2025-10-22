# Methodology (Paper Scaffold)

We cast multi-period budget allocation as a multi-objective RL problem.
- **State**: department-wise rolling KPIs (mean ROI, variance, last allocation), global budget, time.
- **Action**: simplex allocation vector over departments (0..1) summing to 1, then scaled by budget.
- **Rewards (vector)**: [Expected NPV ↑, Risk ↓, Fairness ↑, Strategy alignment ↑].
- **Scalarization**: weighted sum for PG baseline; evolutionary sampler for Pareto exploration.
- **Constraints**: min/max per unit integrated via soft penalties and projection.
- **Risk**: variance proxy and CVaR-style tail penalty on simulated returns.
