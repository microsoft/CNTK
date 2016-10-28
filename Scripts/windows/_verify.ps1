#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

function VerifyOperations()
{
    Write-Host "Determining Operations to perform"

    foreach ($item in $operations) {
        $needsInstall = $false

        foreach ($verificationItem in $item.Verification) {
            
            $needsInstall = VerifyItem $verificationItem
            if (-not $needsInstall) {
                $global:operationList += $item
                break
            }
        }
    }

    Write-Host 

    if ($global:operationList.Count -gt 0) {
        Write-Host "The following operations will be performed:"

        foreach ($item in $global:operationList) {
            $info = $item.Info
            Write-Host " * $info"
        }
        
        Write-Host 
        Write-Host "Do you want to continue? (y/n)"
        
        $choice = GetKey '^[yYnN]+$'

        if ($choice -contains "y") {
            return $true
        }
    }
    else {
        Write-Host "No additional installation required"
    }
    return $false
}

function VerifyItem(
    [hashtable] $item
){
    $func = $item["Function"]
    $name = $item["Name"]

    $expr = $func +' $item' 
        
    Write-Verbose "Calling Operation: [$func]: [$name]"
    $result = Invoke-Expression $expr 

    return $result
}

function VerifyInstallationContent(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table

    $func = $table["Function"]
    $path = $table["Path"]

    $result = (join-path $path cntk\cntk.exe | test-path -PathType Leaf) 
    $result = (join-path $path prerequisites\VS2012\vcredist_x64.exe | test-path -PathType Leaf) -and $result
    $result = (join-path $path prerequisites\VS2013\vcredist_x64.exe | test-path -PathType Leaf) -and $result
    $result = (join-path $path prerequisites\MSMpiSetup.exe | test-path -PathType Leaf) -and $result

    if ($result) {
        Write-Verbose "[$func]: [$path] returned [$result]"
        return $result
    }
    
    throw "`nFatal Error: Files from CNTK binary download package are missing!`nThe install script must be run out of the unpacked binary CNTK package, not from a CNTK source clone."
}

function VerifyDirectory(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table

    $func = $table["Function"]
    $path = $table["Path"]

    $result = (test-path -path $path -PathType Container)

    Write-Verbose "[$func]: [$path] returned [$result]"
    
    return $result
}

function VerifyWheelDirectory(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table

    $func = $table["Function"]
    $path = $table["WheelDirectory"]
    $forceUpdate = $table["ForceUpdate"]

    if ($forceUpdate) {
        $result = $false
    }
    else {
        $result = (test-path -path $path -PathType Container)
    }

    Write-Verbose "[$func]: [$path] returned [$result]"
    
    return $result
}

function VerifyFile(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table

    $func = $table["Function"]
    $path = $table["Path"]

    $result = (test-path -path $path -PathType Leaf)

    Write-Verbose "[$func]: [$path] returned [$result]"
    
    return $result
}

function VerifyRegistryKey(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table

    $func = $table["Function"]
    $key = $table["Key"]

    $result = (test-path -path $key)

    Write-Verbose "[$func]: [$key] returned [$result]"
    
    return $result
}

function VerifyRegistryKeyName(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table

    $func     = $table["Function"]
    $key      = $table["Key"]
    $regName  = $table["RegName"]

    $result = Test-ItemProperty -Path $key -Name $regName

    Write-Verbose "[$func]: [$key]:[$regname] returned [$result]"
    
    return $result
}

function VerifyRegistryKeyNameData(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table

    $func     = $table["Function"]
    $key      = $table["Key"]
    $regName  = $table["RegName"]
    $regData  = $table["RegData"]

    $result = (test-path -path $key)

    if ($result) {
        $theKeyObj = get-item $key
        $result = ($theKeyObj.GetValue("$regName") -eq $regData)
    }

    Write-Verbose "[$func]: [$key]:[$regname] == [$regData] returned [$result]"
    
    return $result
}

function Test-ItemProperty (
    [string] $Path, 
    [string] $Name
)
{
    if (Test-Path $Path) {
        try {
            $ItemProperty = Get-ItemProperty -Path $Path -Name $Name -ErrorAction SilentlyContinue
            if ( $ItemProperty -ne $null ) {
                return $true 
            }
        }
        catch {
            return $false
        }
    }
    return $false
}