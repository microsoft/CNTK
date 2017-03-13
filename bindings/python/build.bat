setlocal

cd "%~dp0"

if not defined VS140COMNTOOLS (
  @echo Environment variable VS140COMNTOOLS not defined.
  @echo Make sure Visual Studion 2015 Update 3 is installed.
  goto FIN
)
set VCDIRECTORY=%VS140COMNTOOLS%
if "%VCDIRECTORY:~-1%"=="\" set VCDIRECTORY=%VCDIRECTORY:~,-1%

if not exist "%VCDIRECTORY%\..\..\VC\vcvarsall.bat" (
  echo Error: "%VCDIRECTORY%\..\..\VC\vcvarsall.bat" not found. 
  echo Make sure you have installed Visual Studion 2015 Update 3 correctly.  
  goto FIN
)

call "%VCDIRECTORY%\..\..\VC\vcvarsall.bat" amd64 

set MSSdk=1
set DISTUTILS_USE_SDK=1

python .\setup.py build_ext --inplace --force --compiler msvc
if errorlevel 1 exit /b 1

set PATH=%CD%\..\..\x64\Release;%PATH%
set PYTHONPATH=%CD%;%CD%\examples;%PYTHONPATH%

pushd cntk\tests
echo RUNNING cntk unit tests...
pytest --deviceid gpu
if errorlevel 1 exit /b 1
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

pushd cntk\logging\tests
echo RUNNING cntk\logging unit tests...
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

pushd cntk\utils\tests
echo RUNNING cntk\utils unit tests...
pytest --deviceid gpu
if errorlevel 1 exit /b 1
echo(
popd

endlocal
