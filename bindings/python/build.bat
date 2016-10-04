setlocal

SET PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC;C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin
SET MSSdk=1
SET DISTUTILS_USE_SDK=1
call vcvarsall amd64

cd cntk\swig
call "swig.bat"

cd ..\..

python .\setup.py build_ext -if -c msvc --plat-name=win-amd64

set PATH=%cd%\..\..\x64\Release;%PATH%
set PYTHONPATH=%cd%;%cd%\examples;%PYTHONPATH%

cd cntk\ops\tests
echo RUNNING cntk\ops unit tests...
pytest
echo(
cd ..\..\..

endlocal
