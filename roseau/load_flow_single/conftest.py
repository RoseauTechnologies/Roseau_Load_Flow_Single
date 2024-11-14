from pathlib import Path

import platformdirs
import pytest
from _pytest.monkeypatch import MonkeyPatch

from roseau.load_flow import activate_license
from roseau.load_flow.utils.log import set_logging_config

HERE = Path(__file__).parent.expanduser().absolute()
TEST_ALL_NETWORKS_DATA_FOLDER = HERE / "tests" / "data" / "networks"

TEST_DGS_NETWORKS = list((HERE / "tests" / "data" / "dgs").rglob("*.json"))
TEST_DGS_NETWORKS_IDS = [x.stem for x in TEST_DGS_NETWORKS]
TEST_DGS_SPECIAL_NETWORKS_DIR = HERE / "tests" / "data" / "dgs" / "special"

THREE_PHASES_TRANSFORMER_TYPES = [
    "Dd0",
    "Dd6",
    "Dyn11",
    "Dyn5",
    "Dzn0",
    "Dzn6",
    "Yd11",
    "Yd5",
    "Yyn0",
    "Yyn6",
    "Yzn11",
    "Yzn5",
]


@pytest.fixture(autouse=True, scope="session")
def _log_setup():
    """A basic fixture (automatically used) to set the log level"""
    set_logging_config(verbosity="debug")


@pytest.fixture(autouse=True, scope="session")
def _license_setup(_log_setup, tmp_path_factory):
    """A basic fixture (automatically used) to activate a license for the tests"""
    license_folderpath = tmp_path_factory.mktemp("roseau-test")

    def _user_cache_dir():
        return str(license_folderpath)

    mpatch = MonkeyPatch()
    mpatch.setattr(target=platformdirs, name="user_cache_dir", value=_user_cache_dir)
    activate_license(key=None)  # Use the environment variable `ROSEAU_LOAD_FLOW_LICENSE_KEY`
    yield
    mpatch.undo()


@pytest.fixture(params=["impedance", "power"], ids=["impedance", "power"])
def network_load_data_name(request) -> str:
    return request.param


@pytest.fixture(params=TEST_DGS_NETWORKS, ids=TEST_DGS_NETWORKS_IDS)
def dgs_network_path(request) -> Path:
    return request.param


@pytest.fixture
def dgs_special_network_dir() -> Path:
    return TEST_DGS_SPECIAL_NETWORKS_DIR


@pytest.fixture
def test_networks_path() -> Path:
    return TEST_ALL_NETWORKS_DATA_FOLDER


@pytest.fixture(params=THREE_PHASES_TRANSFORMER_TYPES, ids=THREE_PHASES_TRANSFORMER_TYPES)
def three_phases_transformer_type(request) -> str:
    return request.param
