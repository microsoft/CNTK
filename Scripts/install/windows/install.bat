@echo off
@REM
@REM Copyright (c) Microsoft. All rights reserved.
@REM Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
@REM

setlocal
set allArgs=%*
pushd "%~dp0"
set psInstall="%~p0\ps\install.ps1"
set PSModulePath=%PSModulePath%;%~dp0\ps\Modules
:loop
if /I "%~1"=="-h" goto :helpMsg
if /I "%~1"=="/h" goto :helpMsg
if /I "%~1"=="-help" goto :helpMsg
if /I "%~1"=="/help" goto :helpMsg
if "%~1"=="-?" goto :helpMsg
if "%~1"=="/?" goto :helpMsg
shift

if not "%~1"=="" goto loop

powershell -NoProfile -NoLogo -Command unblock-file ps\*.ps1
powershell -NoProfile -NoLogo -Command unblock-file ps\Modules\Download\*.psm1
popd
powershell -NoProfile -NoLogo -ExecutionPolicy RemoteSigned %psInstall% %allArgs%

goto :EOF

:helpMsg
@echo.
@echo. CNTK Binary Install Script
@echo.
@echo More help can be found at: 
@echo     https://github.com/Microsoft/CNTK/wiki/Setup-Windows-Binary-Script
@echo.
@echo An overview of the available command line options can be found at:
@echo     https://github.com/Microsoft/CNTK/wiki/Setup-Windows-Binary-Script-Options
@echo.

endlocal
