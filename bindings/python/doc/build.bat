REM Steps to recreate the docs:

setlocal

cd "%~dp0"
set PYTHONPATH=%CD%\..
echo PYTHONPATH=%PYTHONPATH%
set PATH=%CD%\..;%CD%\..\..\..\x64\Release;%PATH%
echo PATH=%PATH%

sphinx-apidoc.exe ..\cntk -o . -f
if errorlevel 1 exit /b 1

.\make.bat html

echo start _build\html\index.html

