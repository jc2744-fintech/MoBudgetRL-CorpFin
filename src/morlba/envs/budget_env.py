import numpy as np

def gini(x):
    x = np.asarray(x, dtype=float)
    if np.amin(x) < 0:
        x -= np.amin(x)
    x += 1e-9
    x = np.sort(x)
    n = x.size
    return (2 * np.arange(1, n+1) - n - 1).dot(x) / (n * x.sum())

class BudgetEnv:
    """Corporate budget allocation environment.

    State: concat of dept ROI means/sigmas, last allocation, time frac.
    Action: allocation vector (softmax) over D departments; projected to constraints.
    Reward (scalar for training): weighted sum of objectives.
    Info carries vector objectives for analysis.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.D = cfg["env"]["departments"]
        self.B = cfg["env"]["budget"]
        self.min_pct = cfg["env"]["min_pct_per_unit"]
        self.max_pct = cfg["env"]["max_pct_per_unit"]
        self.synergy = cfg["env"]["synergy_scale"]
        self.roi_mu_range = cfg["env"]["roi_mu_range"]
        self.roi_sigma_range = cfg["env"]["roi_sigma_range"]
        self.strategy = np.array(cfg["env"]["strategy_scores"][:self.D])
        self.w = cfg["objectives"]["weights"]
        self.gamma = 0.99
        self.reset()

    def reset(self):
        self.t = 0
        self.horizon = self.cfg["horizon"]
        self.mu = np.random.uniform(self.roi_mu_range[0], self.roi_mu_range[1], size=self.D)
        self.sigma = np.random.uniform(self.roi_sigma_range[0], self.roi_sigma_range[1], size=self.D)
        self.last_a = np.ones(self.D)/self.D
        return self._state()

    def _state(self):
        time_frac = self.t / max(1, self.horizon)
        return np.concatenate([self.mu, self.sigma, self.last_a, [time_frac]])

    def _project(self, a):
        # Simple projection to [min,max] and sum=1
        a = np.maximum(a, 1e-6)
        a = a / a.sum()
        a = np.clip(a, self.min_pct, self.max_pct)
        a = a / a.sum()
        return a

    def step(self, a):
        a = self._project(a)
        # Returns per dept ~ Normal(mu, sigma) scaled by allocation and budget
        noise = np.random.normal(0, self.sigma, size=self.D)
        dept_ret = self.mu + noise
        # Synergy: quadratic bonus for diversification
        div_bonus = self.synergy * (1 - np.sum(a**2))
        total_return = (a * dept_ret).sum() + div_bonus

        # Vector objectives (higher is better for all except risk which we negate variance)
        expected = float((a * self.mu).sum())
        risk = - float((a**2 * self.sigma**2).sum())  # negative variance as "higher better"
        fairness = float(1 - gini(a))                 # 1-Gini in [0,1]
        strategy_align = float((a * self.strategy).sum())

        # Scalarization
        r = (self.w["npv"] * expected
             + self.w["risk"] * risk
             + self.w["fairness"] * fairness
             + self.w["strategy"] * strategy_align)

        self.last_a = a
        self.t += 1
        done = self.t >= self.horizon
        s2 = self._state()
        info = {"expected": expected, "risk": risk, "fairness": fairness, "strategy": strategy_align}
        return s2, float(r), done, info
