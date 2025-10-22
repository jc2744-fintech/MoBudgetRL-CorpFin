from pathlib import Path
from morlba.cli import demo

def test_demo(tmp_path: Path):
    out = tmp_path / "artifacts"
    demo.callback = None  # Typer appeasement
    demo(episodes=5, out=str(out))
    assert (out / "summary.json").exists()
    assert (out / "frontier.json").exists()
