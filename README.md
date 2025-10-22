# MoBudgetRL-CorpFin

**Short description**: Multi-objective reinforcement learning framework for **corporate budget allocation** that optimizes **return, risk (CVaR/variance), and fairness across business units**, with strategic constraints and Pareto analysis. Includes a configurable simulator, scalarization-based PG agent, evolutionary baseline, reproducible configs, tests, and CI.

## Highlights
- Stochastic corporate finance simulator with departments, project pipelines, and synergies
- Objectives: **Expected NPV**, **Risk (variance/CVaR)**, **Fairness (allocation Gini)**, **Strategic Score**
- Constraints: budget cap, min/max per unit, strategic floors, soft penalties
- Agents: **Policy Gradient (weighted scalarization)** + **Evolutionary (Pareto sampler)** baseline
- Pareto frontier estimation + metric dashboards (JSON artifacts)
- Clean, offline demo; plug in deep RL later

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tiny demo (no external APIs)
python -m morlba.cli demo --episodes 20 --out artifacts/demo

# Run tests
pytest -q
```

## Structure
```
MoBudgetRL-CorpFin/
├─ src/morlba/               # Core package (env, agents, eval, utils)
├─ configs/                  # Experiment configs
├─ data/synthetic/           # Synthetic strategy/ROI priors
├─ docs/                     # Paper scaffolding & design docs
├─ tests/                    # Unit/integration tests
├─ .github/workflows/        # CI
└─ artifacts/                # Outputs (gitignored)
```
License: MIT © 2025
