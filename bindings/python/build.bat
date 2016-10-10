setlocal

cd "%~dp0"

call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall" amd64

set MSSdk=1
set DISTUTILS_USE_SDK=1

python .\setup.py build_ext --inplace --force
if errorlevel 1 exit /b 1

set PATH=%CD%\..\..\x64\Release;%PATH%
set PYTHONPATH=%CD%;%CD%\examples;%PYTHONPATH%

pushd cntk\tests
echo RUNNING cntk unit tests...
pytest --deviceid gpu
if errorlevel 1 exit /b 1
echo(
popd

pushd cntk\ops\tests
echo RUNNING cntk\ops unit tests...
pytest --deviceid gpu
if errorlevel 1 exit /b 1
echo(
popd

endlocal
