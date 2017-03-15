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

function InstallExe(
    [Parameter(Mandatory = $true)][string] $cmd,
    [Parameter(Mandatory = $true)][string] $param,
    [string] $dir = $null,
    [string] $platform = $null,
    [bool] $runAs = $false,
    [integer] $maxErrorLevel = 0,
    [string] $message= $null)
{
    if ($platform -ne $null) {
        $runningOn = ((Get-WmiObject -class Win32_OperatingSystem).Caption).ToUpper()
        $platform  = ($platform.ToString()).ToUpper()

        if (-not $runningOn.StartsWith($platform)) {
            return
        }
    }

    if ($message) {
        Write-Host $message
    }
    
    if ($dir) {
        DoProcess -command $cmd -param $param -requiresRunAs $runAs -workingDir $dir -maxErrorLevel $maxErrorLevel
    }
    else {
        DoProcess -command $cmd -param $param -requiresRunAs $runAs -maxErrorLevel $maxErrorLevel
    }
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
    $installExeParam = @{ cmd = "$basepath\Scripts\conda.exe"; param = "env $condaOp --file `"$ymlFile`" --name `"$targetDir`""; dir = "$basePath\Scripts" }
    InstallExe $installExeParam
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
    if (-not $Execute) {
         Write-Host  "** Running in DEMOMODE - setting Exit Code **: 0"
         return 
    }
    $condaExe = Join-Path $basePath 'Scripts\conda.exe'
    $newPaths = Invoke-DosCommand $condaExe (Write-Output ..activate cmd.exe $envName)  -maxErrorLevel 0

    $oldPath = $env:PATH
    $env:PATH = $newPaths + ';' + $env:PATH

    Invoke-DosCommand pip (Write-Output install $whlUrl) -maxErrorLevel 0
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


function DoProcess(
    [string]  $command,
    [string]  $param,
    [string]  $workingDir = "",
    [boolean] $requiresRunAs = $false,
    [int] $maxErrorLevel)
{
    $info = "start-process [$command] with [$param]"

    Write-Verbose "$info"

    if (-not $Execute) {
         Write-Host  "** Running in DEMOMODE - setting Exit Code **: 0"
         return
    }

    if ($workingDir.Length -eq 0) {
        if ($requiresRunAs) {
            $process = start-process -FilePath "$command" -ArgumentList "$param" -Wait -PassThru -Verb runas
        }
        else {
            $process = start-process -FilePath "$command" -ArgumentList "$param" -Wait -PassThru
        }

    }
    else {
        if ($requiresRunAs) {
            $process = start-process -FilePath "$command" -ArgumentList "$param" -Wait -PassThru -Verb runas -WorkingDirectory "$workingDir"
        }
        else {
            $process = start-process -FilePath "$command" -ArgumentList "$param" -Wait -PassThru -WorkingDirectory "$workingDir"
        }
    }

    $eCode = ($process.ExitCode)

    if ($ecode -gt $maxErrorLevel) {
        throw "Running 'start-process $command $param' failed with exit code [$ecode]"
    }
    
    return
}

function Invoke-DosCommand {
  [CmdletBinding()]
  Param(
    [ValidateScript({ Get-Command $_ })]
    [string] $Command,
    [string[]] $Argument,
    [string] [ValidateScript({ Test-Path -PathType Container $_ })] $WorkingDirectory,
    [int] $maxErrorLevel,
    [switch] $SuppressOutput
  )
    Write-Verbose "Running '$Command $Argument'"
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