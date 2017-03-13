# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

# Assumes clean Git working directory
[CmdletBinding()]
Param([Parameter(Mandatory=$true)][string]$Output)

if (-not (Get-Command -ErrorAction SilentlyContinue 7za.exe)) {
    throw "7za.exe not found in path"
}

Set-StrictMode -Version Latest

$ErrorActionPreference = 'Stop'

$isVerbose = $PSBoundParameters.ContainsKey('Verbose')

# Normalize path
$Output = [System.IO.Path]::Combine((Get-Location).Path, $Output)
$Output = [System.IO.Path]::GetFullPath($Output)

# N.B. explicit -Verbose conflicts with implicit $ErrorActionPreference... (also below)
Remove-Item -ErrorAction SilentlyContinue -Verbose:$isVerbose -Recurse SamplesZip, $Output

$null = New-Item -ItemType Directory SamplesZip

Copy-Item -ErrorAction Stop -Verbose:$isVerbose -Path LICENSE.md -Destination SamplesZip\LICENSE.md

Copy-Item -ErrorAction Stop -Verbose:$isVerbose -Path Scripts\install\sample_requirements.txt -Destination SamplesZip\requirements.txt

Copy-Item -ErrorAction Stop -Verbose:$isVerbose -Recurse Examples, Tutorials -Destination SamplesZip

Push-Location SamplesZip

try {
    7za.exe a -bd $Output .
    if ($LASTEXITCODE -ne 0) {
        throw "7za.exe returned non-zero exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
