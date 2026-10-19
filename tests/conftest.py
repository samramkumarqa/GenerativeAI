import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

SAMPLE_BRD = ROOT / "samples" / "sample_brd.pdf"


@pytest.fixture
def sample_text():
    from tcgen.loaders import read_upload

    return read_upload(SAMPLE_BRD.name, SAMPLE_BRD.read_bytes())
