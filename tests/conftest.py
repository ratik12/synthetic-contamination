import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    from tests.helpers import build_tiny_models
    return build_tiny_models(tmp_path_factory.mktemp("tiny"))
