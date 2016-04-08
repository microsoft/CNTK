import sys

collect_ignore = ["setup.py"]

# content of conftest.py

def pytest_addoption(parser):
    parser.addoption("--deviceid", action="append", default=[],
        help="list of device ids to pass to test functions")

def pytest_generate_tests(metafunc):    
    if 'device_id' in metafunc.fixturenames:        
        metafunc.parametrize("device_id",
                             metafunc.config.option.deviceid)