from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def test_there_are_notebooks() -> None:
    assert NOTEBOOKS, "no notebook found"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs_end_to_end(
    path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every notebook ends with asserts on its numbers: running it top to bottom checks them.
    # Figures go to a temporary folder so that the test never rewrites the committed PNGs.
    monkeypatch.setenv("PAIRS_RESULTS_DIR", str(tmp_path))
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
