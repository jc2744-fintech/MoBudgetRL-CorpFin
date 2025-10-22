from typer import Typer, Option
from rich import print
from pathlib import Path
from .utils.config import load_yaml, set_seed
from .envs.budget_env import BudgetEnv
from .agents.pg import PGAgent
from .agents.evo import EvoSampler
from .eval.frontier import evaluate_policy, estimate_frontier

app = Typer(help="MoBudgetRL Corporate Finance CLI")

@app.command()
def demo(episodes: int = Option(30, help="Training episodes"),
         out: str = Option("artifacts/demo", help="Output directory"),
         config: str = Option("configs/experiment.yaml", help="Config path")):
    cfg = load_yaml(config)
    set_seed(cfg.get("seed", 42))
    env = BudgetEnv(cfg)
    algo = cfg["agent"]["algo"]
    agent = PGAgent(env, cfg) if algo == "pg" else EvoSampler(env, cfg)

    Path(out).mkdir(parents=True, exist_ok=True)

    returns = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        G = 0.0
        while not done:
            a = agent.act(s)
            s2, r, done, info = env.step(a)
            agent.observe(s, a, r, s2, done)
            s = s2
            G += r
        agent.end_episode()
        returns.append(G)
        if (ep+1) % max(1, cfg["logging"]["every"]) == 0:
            print(f"[cyan]Episode {ep+1}[/cyan] scalarized return: {G:.3f}")

    summary = evaluate_policy(env, agent, episodes=cfg["eval"]["episodes"])
    (Path(out) / "summary.json").write_text(__import__("json").dumps(summary, indent=2))

    # Frontier (randomized policies baseline)
    frontier = estimate_frontier(env, samples=cfg["eval"]["pareto_samples"])
    (Path(out) / "frontier.json").write_text(__import__("json").dumps(frontier, indent=2))

    print(f"[bold green]Done.[/bold green] Artifacts in {out}")

if __name__ == "__main__":
    app()
