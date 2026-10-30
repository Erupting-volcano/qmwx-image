"""Test actual browser script with synthetic data, never a paid model call."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("scenario", [
    "sourced_candidates", "conflicts_and_partial", "old_and_disabled", "failed_latest", "foreign_sources", "privacy_config",
])
def test_understanding_feedback_sources_and_safe_dom(scenario):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Optional DOM checks require local Node.js")
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([node, str(root / "tests" / "object_understanding_web_dom.cjs"), scenario],
                               cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "understanding DOM checks passed" in completed.stdout
