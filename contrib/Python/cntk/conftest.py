import sys

collect_ignore = ["setup.py"]

# content of conftest.py
_DEFAULT_DEVICE_ID=-1

def pytest_addoption(parser):
    parser.addoption("--deviceid", action="append", default=[_DEFAULT_DEVICE_ID],
        help="list of device ids to pass to test functions")

def pytest_generate_tests(metafunc):    
    if 'device_id' in metafunc.fixturenames:        
        if (len(metafunc.config.option.deviceid)) > 1:
            del metafunc.config.option.deviceid[0]
            
        devices = set()
        for d_id in metafunc.config.option.deviceid:
            try:
                devices.add(int(d_id))
            except ValueError:
                raise RuntimeError("invalid deviceid value {0}, please use integer values".format(d_id))
                    
        metafunc.parametrize("device_id", devices)