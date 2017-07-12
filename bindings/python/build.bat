setlocal

cd "%~dp0"

if not defined VS140COMNTOOLS (
  @echo Environment variable VS140COMNTOOLS not defined.
  @echo Make sure Visual Studion 2015 Update 3 is installed.
  exit /b 0
)

set VCDIRECTORY=%VS140COMNTOOLS%
if "%VCDIRECTORY:~-1%"=="\" set VCDIRECTORY=%VCDIRECTORY:~,-1%

if not exist "%VCDIRECTORY%\..\..\VC\vcvarsall.bat" (
  echo Error: "%VCDIRECTORY%\..\..\VC\vcvarsall.bat" not found. 
  echo Make sure you have installed Visual Studion 2015 Update 3 correctly.  
  exit /b 0
)

call "%VCDIRECTORY%\..\..\VC\vcvarsall.bat" amd64 

set MSSdk=1
set DISTUTILS_USE_SDK=1

set CNTK_COMPONENT_VERSION=2.0

if defined SWIG_PATH set PATH=%PATH%;%SWIG_PATH%

python .\setup.py build_ext --inplace --force --compiler msvc && ^
cd cntk && ^
pytest --deviceid gpu
