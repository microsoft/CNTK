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
Move-Item -Path InstallCache -Destination $installCache

.\BinaryDrop\cntk\Scripts\install\windows\install.ps1 -NoConfirm

Set-Location BinaryDrop
..\test-install.bat cntk\scripts\cntkpy35.bat
if ($LASTEXITCODE -ne 0) {
  throw "Fail"
}
