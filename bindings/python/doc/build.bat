setlocal enabledelayedexpansion

cd /d "%~dp0"
if not defined SPHINX_DOCFX_BUILD (
  @REM Sphinx DocFx build should go against _installed_ module,
  @REM since otherwise file paths are broken in output.
  set PYTHONPATH=!CD!\..
  set PATH=!CD!\..;!CD!\..\..\..\x64\Release;!PATH!
)

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

.\make.bat html
if errorlevel 1 exit /b 1

echo start _build\html\index.html
