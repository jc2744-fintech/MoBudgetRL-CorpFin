import yaml, numpy as np, random
def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
def set_seed(seed: int):
    np.random.seed(seed); random.seed(seed)
