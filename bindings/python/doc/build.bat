setlocal

cd /d "%~dp0"
set PYTHONPATH=%CD%\..
set PATH=%CD%\..;%CD%\..\..\..\x64\Release;%PATH%

@REM TODO better align conf.py exclude with excluded paths here
sphinx-apidoc.exe ..\cntk -o . -f ^
  ..\cntk\tests ^
  ..\cntk\debugging\tests ^
  ..\cntk\internal\tests ^
  ..\cntk\io\tests ^
  ..\cntk\layers\tests ^
  ..\cntk\learners\tests ^
  ..\cntk\logging\tests ^
  ..\cntk\losses\tests ^
  ..\cntk\metrics\tests ^
  ..\cntk\ops\tests ^
  ..\cntk\train\tests ^
  ..\cntk\utils\tests

if errorlevel 1 exit /b 1

.\make.bat html
if errorlevel 1 exit /b 1

echo start _build\html\index.html
