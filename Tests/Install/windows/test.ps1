# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
[CmdletBinding()]
Param([string]$WheelBaseUrl)

$image = 'cntk:installtest'

$serverInfo = docker version --format '{{json .Server}}' | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) {
  throw "Fail"
}

$expectedOsArch = 'windows/amd64'
if (("{0}/{1}" -f $serverInfo.Os, $serverInfo.Arch) -ne $expectedOsArch) {
  throw "docker server OS/Arch is different from $expectedOsArch. Make sure to switch to Windows Containers."
}

docker build -t $image .
if ($LASTEXITCODE -ne 0) {
  throw "Fail"
}
docker run --rm $image powershell c:/local/test-install.ps1 @PSBoundParameters
if ($LASTEXITCODE -ne 0) {
  throw "Fail"
}
docker rmi $image
# Ignore error here
