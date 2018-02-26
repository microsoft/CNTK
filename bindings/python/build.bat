setlocal

cd "%~dp0"

if not defined VS2017INSTALLDIR (
  @echo Environment variable VS2017INSTALLDIR not defined.
  @echo Make sure Visual Studion 2017 is installed.
  goto FIN
)

if not exist "%VS2017INSTALLDIR%\VC\Auxiliary\build\vcvarsall.bat" (
  echo Error: "%VS2017INSTALLDIR%\VC\Auxiliary\build\vcvarsall.bat" not found.
  echo Make sure you have installed Visual Studion 2017 correctly.
  goto FIN
)

call "%VS2017INSTALLDIR%\VC\Auxiliary\build\vcvarsall.bat" amd64 -vcvars_ver=14.11

set MSSdk=1
set DISTUTILS_USE_SDK=1
set CNTK_VERSION=2.4
set CNTK_VERSION_BANNER=%CNTK_VERSION%+
set CNTK_COMPONENT_VERSION=%CNTK_VERSION%

python .\setup.py build_ext --inplace --force --compiler msvc
if errorlevel 1 exit /b 1

set PATH=%CD%\..\..\x64\Release;%PATH%
set PYTHONPATH=%CD%;%CD%\..\..\Scripts;%CD%\examples;%PYTHONPATH%

pushd cntk\tests
echo RUNNING cntk unit tests...
pytest --deviceid gpu
if errorlevel 1 exit /b 1
echo(
popd

pushd cntk\debugging\tests
echo RUNNING cntk\debugging unit tests...
pytest --deviceid gpu
if errorlevel 1 exit /b 1
echo(
popd

pushd cntk\internal\tests
echo RUNNING cntk\internal unit tests...
pytest --deviceid gpu
echo(
popd

pushd cntk\io\tests
echo RUNNING cntk\io unit tests...
pytest --deviceid gpu
if errorlevel 1 exit /b 1
echo(
popd

pushd cntk\layers\tests
echo RUNNING cntk\layers unit tests...
pytest --deviceid gpu
if errorlevel 1 exit /b 1
echo(
popd

pushd cntk\learners\tests
echo RUNNING cntk\learners unit tests...
pytest --deviceid gpu
if errorlevel 1 exit /b 1
echo(
popd

pushd cntk\logging\tests
echo RUNNING cntk\logging unit tests...
pytest --deviceid gpu
if errorlevel 1 exit /b 1
echo(
popd

pushd cntk\losses\tests
echo RUNNING cntk\losses unit tests...
pytest --deviceid gpu
if errorlevel 1 exit /b 1
echo(
popd

pushd cntk\metrics\tests
echo RUNNING cntk\metrics unit tests...
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

pushd cntk\train\tests
echo RUNNING cntk\train unit tests...
pytest --deviceid gpu
if errorlevel 1 exit /b 1
echo(
popd

pushd cntk
echo RUNNING cntk\variables doctests...
pytest variables.py
if errorlevel 1 exit /b 1
echo(
popd

pushd cntk\losses
echo RUNNING cntk\losses doctests...
pytest __init__.py
if errorlevel 1 exit /b 1
echo(
popd

pushd cntk\ops
echo RUNNING cntk\ops doctests...
pytest __init__.py
if errorlevel 1 exit /b 1
echo(
popd

pushd cntk\ops
echo RUNNING cntk\ops function doctests...
pytest functions.py
if errorlevel 1 exit /b 1
echo(
popd

pushd cntk\ops\sequence
echo RUNNING cntk\ops\sequence doctests...
pytest __init__.py
if errorlevel 1 exit /b 1
echo(
popd

pushd cntk\layers
echo RUNNING cntk\blocks doctests...
pytest blocks.py
if errorlevel 1 exit /b 1
echo(
popd

pushd cntk\layers
echo RUNNING cntk\layers doctests...
pytest layers.py
if errorlevel 1 exit /b 1
echo(
popd

pushd cntk\layers
echo RUNNING cntk\sequence doctests...
pytest sequence.py
if errorlevel 1 exit /b 1
echo(
popd

pushd cntk\layers
echo RUNNING cntk\higher_order_layers doctests...
pytest higher_order_layers.py
if errorlevel 1 exit /b 1
echo(
popd

endlocal
