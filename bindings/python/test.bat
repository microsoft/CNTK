REM Copyright (c) Microsoft. All rights reserved.
REM
REM Licensed under the MIT license. See LICENSE.md file in the project root
REM for full license information.
REM ==============================================================================

REM This script is used to run tests from a source checkout against local CNTK
REM module and examples.
REM Assumes you are already in an activated CNTK python environment.
REM
REM Note: this is close but not exactly the same as our CI tests. If you need
REM exactly the same setup, install the CNTK python module from a wheel into a
REM clean CNTK python environment, don't modify PYTHONPATH, are run the
REM associated end-to-end tests via TestDriver.py.
REM
REM Note: currently hard-coded for GPU build and testing on GPU device.

setlocal

cd "%~dp0"

set PATH=%CD%\..\..\x64\Release;%PATH%
set PYTHONPATH=%CD%;%PYTHONPATH%

for %%d in (cntk doc examples) do (
  pushd %%d
  echo Testing %%d on GPU...
  pytest --deviceid gpu
  if errorlevel 1 exit /b 1
  echo.
  popd
)

endlocal
