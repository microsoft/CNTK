#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

function ActionOperations(
    [Parameter(Mandatory = $true)][hashtable[]] $operationList)
{
    Write-Host "Performing install operations"

    foreach ($item in $operationList) {
        
        foreach ($actionItem in $item.Action) {
            $params = $actionItem.Params
            $expr = $actionItem.Function +' @params' 
        
            Write-Verbose "Calling Operation: [$expr]($params)"
            Invoke-Expression $expr
        }
    }

    Write-Host "Install operations finished"
    Write-Host
}

function RunDos(
    [Parameter(Mandatory = $true)][string] $cmd,
    [Parameter(Mandatory = $true)][string[]] $param,
    [string] $workingDir = $(Get-Location),
    [int] $maxErrorLevel = 0,
    [string] $platform = $null,
    [string] $message= $null,
    [switch] $noExecute)
{
    Write-Verbose "RunDos '$cmd $param'"
    if ($platform) {
        $runningOn = ((Get-WmiObject -class Win32_OperatingSystem).Caption).ToUpper()
        $platform  = ($platform.ToString()).ToUpper()

        if (-not $runningOn.StartsWith($platform)) {
            Write-Verbose "No platform match [$runningOn] : [$platform]"
            return
        }
    }
    if ($message) {
        Write-Host $message
    }

    Start-DosProcess -command $cmd -argumentList $param -maxErrorLevel $maxErrorLevel -workingDir $workingDir -noExecute:$noExecute
}

function InstallYml(
    [Parameter(Mandatory = $true)][string] $basePath,
    [Parameter(Mandatory = $true)][string] $env,
    [Parameter(Mandatory = $true)][string] $ymlFile,
    [Parameter(Mandatory = $true)][string] $pyVersion,
    [string] $message= $null,
    [switch] $noExecute)
{
    if ($message) {
        Write-Host $message
    }
    $envsDir = Join-Path $basePath envs
    $targetDir = Join-Path $envsDir $env
    $scriptDir = Join-Path  $basePath Scripts
    $condaExe = Join-Path $scriptDir "conda.exe"

    $condaOp = "create"
    $targetDirParam = "--prefix"
    if (Test-Path -path $targetDir -PathType Container) {
        $condaOp = "update"
        $targetDirParam = "--name"
    }

    $param = (Write-Output env $condaOp --file $ymlFile $targetDirParam $targetDir)
    Start-DosProcess -command $condaExe -argumentList $param -workingDir $scriptDir -noExecute:$noExecute
}

function InstallWheel(
    [Parameter(Mandatory = $true)][string] $basePath,
    [Parameter(Mandatory = $true)][string] $envName,
    [Parameter(Mandatory = $true)][string] $whlUrl,
    [string] $message = $null,
    [switch] $noExecute)
{
    if ($message) {
        Write-Host $message
    }
    if ($noExecute) {
         Write-Host  "** Running in DEMOMODE - setting Exit Code **: 0"
         return 
    }
    $condaExe = Join-Path $basePath 'Scripts\conda.exe'
    $newPaths = Invoke-DosCommand $condaExe (Write-Output ..activate cmd.exe $envName) -maxErrorLevel 0

    $oldPath = $env:PATH
    $env:PATH = $newPaths + ';' + $env:PATH

    Start-DosProcess -command "pip" -argumentList (Write-Output install $whlUrl) -noExecute:$noExecute
    $env:PATH = $oldPath 
}

function CreateBatch(
    [Parameter(Mandatory = $true)][string] $filename,
    [Parameter(Mandatory = $true)][string] $pyVersion,
    [Parameter(Mandatory = $true)][string] $basePath,
    [Parameter(Mandatory = $true)][string] $rootDir,
    [switch] $noExecute)
{
    if ($noExecute) {
        Write-Host "Create-Batch [$filename]:No-Execute flag. No file created"
        return
    }

    Remove-Item -Path $filename -ErrorAction SilentlyContinue | Out-Null

    $batchScript = @"
@echo off
if /I "%CMDCMDLINE%" neq ""%COMSPEC%" " (
    echo.
    echo Please execute this script from inside a regular Windows command prompt.
    echo.
    exit /b 0
)
set PATH=$rootDir\cntk;%PATH%
"$anacondaBasePath\Scripts\activate" "$anacondaBasePath\envs\cntk-py$pyVersion"
"@

    add-content -Path $filename -Encoding Ascii -Value $batchScript
}

# vim:set expandtab shiftwidth=4 tabstop=4: