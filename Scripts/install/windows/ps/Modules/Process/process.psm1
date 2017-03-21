#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

Set-StrictMode -Version Latest

function Start-DosProcess{
    [CmdletBinding()]
    Param(
        [Parameter(Mandatory=$true)][string] $command,
        [string[]] $argumentList = @(),
        [string] $workingDir = $(Get-Location),
        [int] $maxErrorLevel = 0,
        [switch] $runAs,
        [switch] $suppressOutput,
        [switch] $checkErrFile,
        [switch] $getOutput,
        [switch] $noExecute)

    Write-Verbose "Start-Process: [$command] Arguments: [$argumentList]"
    
    $params = @{ }
    $errorFile = $null
    $errorContent = $null
    $outputFile = $null
    $outputContent = $null

    if ($noExecute) {
         Write-Host  "** Execute option not set: [$command] - setting Exit Code: 0"
         return
    }

    if (-not (Test-Path $workingDir -PathType Container)) {
        Throw "Start-DosProcess [$command]: not a directory [$workingDir]"
    }

    Push-Location $workingDir -ErrorAction Stop
    
    if (-not (Get-Command $command)) {
        Pop-Location
        Throw "Start-DosProcess [$command]: Unknown command"
    }
    
    if ($argumentList) {
        $params += @{ ArgumentList = $argumentList }
    }
    if ($suppressOutput) {
        $params += @{ WindowStyle = "Hidden" }
    }
    if ($checkErrFile) {
        $errorFile = Get-TempFileName
        $params += @{ RedirectStandardError = $errorFile }
    }
    if ($getOutput) {
        $outputFile = Get-TempFileName
        $params += @{ RedirectStandardOutput = $outputFile }
    }
    if ($runAs) {
        $params += @{ Verb = "runAs" }
    }

    $process = start-process -Wait -PassThru -FilePath $command @params
    
    $eCode = ($process.ExitCode)

    if ($errorFile) {
        $errorContent = Get-Content -Path $errorFile -Encoding Ascii -ErrorAction SilentlyContinue
        Remove-Item $errorFile -ErrorAction SilentlyContinue | Out-Null
    }
    if ($outputFile) {
        $outputContent = Get-Content -Path $outputFile -Encoding Ascii -ErrorAction SilentlyContinue
        Remove-Item $outputFile -ErrorAction SilentlyContinue | Out-Null
    }

    Pop-Location

    if ($ecode -gt $maxErrorLevel) {
        throw "Start-DosProcess [$command]: Arguments: [$argumentList]`nFailed with exit code [$ecode]"
    }
    if ($errorContent) {
        throw "Start-DosProcess [$command]: Arguments: [$argumentList]`nErrors reported:`n$errorContent"
    }
    if ($getOutput) {
        return $outputContent
    }
    
    return
}

Export-ModuleMember -Function (Write-Output `
    Start-DosProcess )

# vim: tabstop=4 shiftwidth=4 expandtab
