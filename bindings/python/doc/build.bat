setlocal

cd /d "%~dp0"
set PYTHONPATH=%CD%\..
set PATH=%CD%\..;%CD%\..\..\..\x64\Release;%PATH%

@REM %SPHINX_APIDOC_OPTIONS%, comma-separated, default members,undoc-members,show-inheritance
@REM do we want undoc-members?
@REM set SPHINX_APIDOC_OPTIONS=members
sphinx-apidoc.exe ..\cntk --module-first --separate --no-toc --output-dir=. --force ^
  ..\cntk\cntk_py.py ^
  ..\cntk\conftest.py ^
  ..\cntk\tests ^
  ..\cntk\debugging\tests ^
  ..\cntk\eval\tests ^
  ..\cntk\internal ^
  ..\cntk\io\tests ^
  ..\cntk\layers\tests ^
  ..\cntk\learners\tests ^
  ..\cntk\logging\tests ^
  ..\cntk\losses\tests ^
  ..\cntk\metrics\tests ^
  ..\cntk\ops\tests ^
  ..\cntk\train\tests

if errorlevel 1 exit /b 1

set SPHINXOPTS=-n %SPHINXOPTS%
call .\make.bat html
if errorlevel 1 exit /b 1
@REM call .\make.bat linkcheck
@REM if errorlevel 1 exit /b 1

echo start _build\html\index.html
