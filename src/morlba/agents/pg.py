import numpy as np

class PGAgent:
    def __init__(self, env, cfg):
        self.env = env
        self.lr = cfg["agent"]["alpha"]
        self.gamma = cfg["agent"]["gamma"]
        self.D = env.D
        self.theta = np.zeros((len(env._state()), self.D))  # linear policy
        self.traj = []

    def _softmax(self, z):
        z = z - np.max(z)
        e = np.exp(z)
        return e / np.sum(e)

    def act(self, s):
        logits = s @ self.theta
        a = self._softmax(logits)
        self.traj.append({"s": s.copy(), "a": a.copy()})
        return a

    def observe(self, s, a, r, s2, done):
        self.traj[-1]["r"] = r

    def end_episode(self):
        # REINFORCE with baseline
        Gs, ret = [], 0.0
        for step in reversed(self.traj):
            ret = step["r"] + self.gamma * ret
            Gs.append(ret)
        Gs = list(reversed(Gs))
        baseline = np.mean(Gs)
        for step, G in zip(self.traj, Gs):
            s = step["s"]; a = step["a"]
            # gradient of softmax linear policy w.r.t theta
            grad_logits = np.outer(s, a)  # approximation (omits full Jacobian for simplicity)
            self.theta += self.lr * (G - baseline) * grad_logits
        self.traj = []
