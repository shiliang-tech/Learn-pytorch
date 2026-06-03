import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
answers = sorted((ROOT / "answers").glob("answer_*.py"))

for file in answers:
    print(f"RUN {file.name}", flush=True)
    result = subprocess.run([sys.executable, str(file)], cwd=ROOT.parent)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

print(f"OK: {len(answers)} answer files ran successfully.")
