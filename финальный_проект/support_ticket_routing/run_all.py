from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).parent


def run(command: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running:", " ".join(command))
    print("=" * 80)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main() -> None:
    run([sys.executable, "prepare_data.py"])
    run([sys.executable, "pipeline.py"])
    run([sys.executable, "eval.py"])
    run([sys.executable, "challenge_eval.py"])

    # LLM judge is optional because it requires a valid token in .env.
    run([sys.executable, "llm_judge.py"])

    print("\nAll project steps finished.")


if __name__ == "__main__":
    main()
