"""Execute the real browser script against a small DOM without web dependencies."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("scenario", ["mixed", "unassessed", "all_failed", "latest_run", "empty_images", "unknown_asset"])
def test_quality_feedback_is_visible_and_uses_safe_dom_text(scenario):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Optional DOM regression checks require a local Node.js executable")
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [node, str(root / "tests" / "quality_web_dom.cjs"), scenario],
        cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=20,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "quality DOM checks passed" in completed.stdout
