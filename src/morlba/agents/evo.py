import numpy as np

class EvoSampler:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg

    def act(self, s):
        # Random simplex sample (Dirichlet)
        a = np.random.dirichlet(alpha=np.ones(self.env.D))
        return a

    def observe(self, *args, **kwargs):
        pass

    def end_episode(self):
        pass
