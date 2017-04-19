# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
[CmdletBinding()]
Param([Parameter(Mandatory=$true)] [string]$PyVersion, [string]$WheelBaseUrl)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

Set-Location c:\local
Expand-Archive -Path BinaryDrop.zip

$installCache = '.\BinaryDrop\cntk\Scripts\install\windows\InstallCache'
Move-Item -Path InstallCache -Destination $installCache

Get-ChildItem .\BinaryDrop\cntk\Scripts\install\windows\ps\ -Recurse -File -Include *.ps1, *.psm1 |
  Add-Content -Stream Zone.Identifier -Value "[ZoneTransfer]`r`nZoneId=3`r`n"

.\BinaryDrop\cntk\Scripts\install\windows\install.bat -NoConfirm @PSBoundParameters
if ($LASTEXITCODE -ne 0) {
  throw "Fail"
}

Set-Location BinaryDrop
..\test-install.bat cntk\scripts\cntkpy$PyVersion.bat
if ($LASTEXITCODE -ne 0) {
  throw "Fail"
}
