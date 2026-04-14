"""
Run the workspace smoke test suite.

Reads `smoke_tests.txt` from the workspace root and `config/build/env_vars.yaml`
for per-script env var overrides, then runs each listed script with the
appropriate environment. Continues through failures and exits non-zero
if any script failed.

Mirrors the logic of the `/smoke-test` skill so CI and local runs stay
in sync.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


WORKSPACE = Path(__file__).resolve().parents[2]
SMOKE_FILE = WORKSPACE / "smoke_tests.txt"
ENV_VARS_FILE = WORKSPACE / "config" / "build" / "env_vars.yaml"
SCRIPTS_DIR = WORKSPACE / "scripts"


def load_smoke_scripts() -> list[str]:
    scripts: list[str] = []
    for line in SMOKE_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        scripts.append(line)
    return scripts


def load_env_config() -> dict:
    if not ENV_VARS_FILE.exists():
        return {"defaults": {}, "overrides": []}
    return yaml.safe_load(ENV_VARS_FILE.read_text()) or {}


def pattern_matches(pattern: str, script_path: str) -> bool:
    if "/" in pattern:
        return pattern in script_path
    return Path(script_path).stem == pattern


def build_env(script_rel: str, cfg: dict) -> dict:
    env = os.environ.copy()
    defaults = cfg.get("defaults") or {}
    env.update({k: str(v) for k, v in defaults.items()})
    for override in cfg.get("overrides") or []:
        if pattern_matches(override["pattern"], script_rel):
            for key in override.get("unset", []):
                env.pop(key, None)
            for key, val in (override.get("set") or {}).items():
                env[key] = str(val)
    return env


def run_one(script_rel: str, cfg: dict) -> tuple[str, int, float, str]:
    env = build_env(script_rel, cfg)
    script_path = SCRIPTS_DIR / script_rel
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(WORKSPACE),
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - t0
    output = result.stdout + result.stderr
    return script_rel, result.returncode, elapsed, output


def main() -> int:
    if not SMOKE_FILE.exists():
        print(f"ERROR: no smoke_tests.txt at {SMOKE_FILE}", file=sys.stderr)
        return 1
    scripts = load_smoke_scripts()
    if not scripts:
        print("No smoke test scripts listed.")
        return 0
    cfg = load_env_config()

    print(f"Running {len(scripts)} smoke test script(s) from {SMOKE_FILE.name}\n")
    failures: list[tuple[str, int, str]] = []
    for script_rel in scripts:
        print(f"::group::{script_rel}")
        name, rc, elapsed, output = run_one(script_rel, cfg)
        print(output, end="")
        status = "PASS" if rc == 0 else f"FAIL (exit {rc})"
        print(f"\n[{status}] {name} — {elapsed:.1f}s")
        print("::endgroup::")
        if rc != 0:
            failures.append((name, rc, output))

    total = len(scripts)
    passed = total - len(failures)
    print(f"\n=== Smoke test summary: {passed}/{total} passed ===")
    for name, rc, _ in failures:
        print(f"  FAIL  {name}  (exit {rc})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
