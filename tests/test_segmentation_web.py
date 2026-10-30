"""Exercise target selection and segmentation rendering using the real page script."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("scenario", [
    "box_coordinates", "pointer_cancel_and_clear", "selection_identity", "oriented_scaled_image",
    "mixed_results", "disabled_and_old_archive", "failed_diagnostic_data",
])
def test_segmentation_selection_and_safe_result_display(scenario):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Optional DOM regression checks require a local Node.js executable")
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [node, str(root / "tests" / "segmentation_web_dom.cjs"), scenario],
        cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=20,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "segmentation DOM checks passed" in completed.stdout
