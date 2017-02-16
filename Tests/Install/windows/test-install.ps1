# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

Set-Location c:\local
Unblock-File -Path BinaryDrop.zip
Expand-Archive -Path BinaryDrop.zip

$installCache = '.\BinaryDrop\cntk\Scripts\install\windows\InstallCache'
New-Item -Type Directory $installCache
Move-Item -Path Anaconda3-4.1.1-Windows-x86_64.exe -Destination $installCache

# Mock host input for installation
function Read-Host { if ($global:readHostMockCtr++) { 'y' } else { '1' } }

.\BinaryDrop\cntk\Scripts\install\windows\install.ps1 -Execute

Set-Location BinaryDrop
..\test-install.bat cntk\scripts\cntkpy35.bat
if ($LASTEXITCODE -ne 0) {
  throw "Fail"
}
