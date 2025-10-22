import numpy as np, json

def rollout(env, policy, episodes=5):
    outs = []
    for _ in range(episodes):
        s = env.reset(); done=False; ret=0.0
        vecs = []
        while not done:
            a = policy.act(s)
            s, r, done, info = env.step(a)
            ret += r
            vecs.append([info["expected"], info["risk"], info["fairness"], info["strategy"]])
        policy.end_episode()
        v = np.mean(np.array(vecs), axis=0).tolist()
        outs.append({"scalar": ret, "vector": {"expected": v[0], "risk": v[1], "fairness": v[2], "strategy": v[3]}})
    return outs

def dominates(a, b):
    return all(x >= y for x,y in zip(a,b)) and any(x > y for x,y in zip(a,b))

def evaluate_policy(env, policy, episodes=5):
    outs = rollout(env, policy, episodes)
    vec = np.mean(np.array([list(o["vector"].values()) for o in outs]), axis=0).tolist()
    return {"episodes": episodes,
            "avg_vector": {"expected": vec[0], "risk": vec[1], "fairness": vec[2], "strategy": vec[3]},
            "avg_scalar": float(np.mean([o["scalar"] for o in outs]))}

def estimate_frontier(env, samples=64):
    # sample random policies to approximate frontier
    vectors = []
    for _ in range(samples):
        env.reset()
        a = np.random.dirichlet(alpha=np.ones(env.D))  # fixed policy (constant alloc)
        # Evaluate fixed allocation over one episode
        s = env.reset(); done=False; vecs=[]
        while not done:
            s, r, done, info = env.step(a)
            vecs.append([info["expected"], info["risk"], info["fairness"], info["strategy"]])
        v = np.mean(np.array(vecs), axis=0).tolist()
        vectors.append(v)
    # extract nondominated set
    nd = []
    for i, vi in enumerate(vectors):
        if not any(dominates(vectors[j], vi) for j in range(samples) if j != i):
            nd.append(vi)
    return {"samples": samples, "nondominated": nd}
