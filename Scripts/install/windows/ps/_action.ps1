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
            $params = $downloadItem.Params
            $expr = $downloadItem.Function +' @params' 
        
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
    [string] $workingDir = $null,
    [integer] $maxErrorLevel = 0,
    [string] $platform = $null,
    [string] $message= $null,
    [switch] $SuppressOutput)
{
    Write-Verbose "RunDos '$Command $Argument'"
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

    $dosParam = @{Command = $cmd; Argument = $param;
                  WorkingDirectory = $workingDir; maxErrorLevel = $maxErrorLevel;
                  SuppressOutput = $SuppressOutput; DontExecute = (-not $Execute) }
    Invoke-DosCommand @dosParam
}

function InstallYml(
    [Parameter(Mandatory = $true)][string] $basePath,
    [Parameter(Mandatory = $true)][string] $env,
    [Parameter(Mandatory = $true)][string] $ymlFile,
    [Parameter(Mandatory = $true)][string] $pyVersion)
{
    $envsDir = Join-Path $basePath envs
    $targetDir = Join-Path $envsDir $env

    $condaOp = "create"
    if (Test-Path -path $targetDir -PathType Container) {
        $condaOp = "update"
    }
    $dosParam = @{Command = "$basepath\Scripts\conda.exe"; Argument = (Write-Output env $condaOp --file `"$ymlFile`" --name `"$targetDir`");
                  maxErrorLevel = 0;
                  SuppressOutput = $false; DontExecute = (-not $Execute) }
    Invoke-DosCommand  @dosParam
}

function InstallWheel(
    [Parameter(Mandatory = $true)][string] $basePath,
    [Parameter(Mandatory = $true)][string] $envName,
    [Parameter(Mandatory = $true)][string] $whlUrl,
    [string] $message = $null)
{
    if ($message) {
        Write-Host $message
    }
    $condaExe = Join-Path $basePath 'Scripts\conda.exe'
    $newPaths = Invoke-DosCommand $condaExe (Write-Output ..activate cmd.exe $envName)  -maxErrorLevel 0 -DontExecute:(-not $Execute)

    $oldPath = $env:PATH
    $env:PATH = $newPaths + ';' + $env:PATH

    Invoke-DosCommand pip (Write-Output install $whlUrl) -maxErrorLevel 0 -DontExecute:(-not $Execute)
    $env:PATH = $oldPath 
}

function CreateBatch(
    [Parameter(Mandatory = $true)][string] $filename,
    [Parameter(Mandatory = $true)][string] $pyVersion,
    [Parameter(Mandatory = $true)][string] $basePath,
    [Parameter(Mandatory = $true)][string] $rootDir)
{
    if (-not $Execute) {
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

function Invoke-DosCommand {
  [CmdletBinding()]
  Param(
    [ValidateScript({ Get-Command $_ })]
    [string] $Command,
    [string[]] $Argument,
    [string] [ValidateScript({ Test-Path -PathType Container $_ })] $WorkingDirectory,
    [int] $maxErrorLevel,
    [switch] $SuppressOutput,
    [switch] $DontExecute
  )
    Write-Verbose "Invoke-DosCommand; Running '$Command $Argument'"
    if ($DontExecute) {
        Write-Host "Execute flag set to $false"
    }
    if ($WorkingDirectory) {
        Push-Location $WorkingDirectory -ErrorAction Stop
    }
    if ($SuppressOutput) {
        $null = & $Command $Argument 2>&1
    } else {
        & $Command $Argument
    }
    if ($WorkingDirectory) {
        Pop-Location
    }
    if ($LASTEXITCODE -gt $maxErrorLevel) {
        throw "Running '$Command $Argument' failed with exit code $LASTEXITCODE"
    }
}

# vim:set expandtab shiftwidth=4 tabstop=4: