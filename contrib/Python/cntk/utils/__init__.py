import os

if "CNTK_EXECUTABLE_PATH" not in os.environ:
    raise ValueError(
        "you need to point environmental variable 'CNTK_EXECUTABLE_PATH' to the CNTK binary")

CNTK_EXECUTABLE_PATH = os.environ['CNTK_EXECUTABLE_PATH']
