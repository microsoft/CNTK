#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#
function VerifyOperations(
    [bool] $NoConfirm)
{
    Write-Host "Determining Operations to perform. This will take a moment..."

    foreach ($item in $operations) {
        $needsInstall = $false

        foreach ($verificationItem in $item.Verification) {
            
            $needsInstall = VerifyItem $verificationItem
            if (-not $needsInstall) {
                $Script:operationList += $item
                break
            }
        }
    }

    Write-Host 

    if ($Script:operationList.Count -gt 0) {
        Write-Host "The following operations will be performed:"

        foreach ($item in $Script:operationList) {
            $info = $item.Info
            Write-Host " * $info"
        }
        if ($NoConfirm) {
            return $true
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
    [hashtable] $item)
{
    $func = $item["Function"]
    $name = $item["Name"]

    $expr = $func +' $item' 
        
    Write-Verbose "Calling Operation: [$func]: [$name]"
    $noInstallRequired = Invoke-Expression $expr 

    return $noInstallRequired
}

function VerifyScanPrograms(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table
    $func = $table["Function"]
    $noInstallRequired = $true
    
    # no actual work is being performed, just the script local datastructure with the list
    # of installed programs is being initialized
    LoadWin32Product
    return $noInstallRequired
}

function VerifyWin32ProductExists(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table
    $func = $table["Function"]
    $match = $table["Match"]
    $noInstallRequired = $true

    $allProducts = LoadWin32Product
    $productList = @($allProducts | Where-Object { $_.Name -match $match } )
    
    if ($productList.Count -eq 0) {
        $noInstallRequired = $false
    }

    Write-Verbose "[$func]: Product [$match] returned [$noInstallRequired]"
    return $noInstallRequired
}

function VerifyWin32ProductVersion(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table
    $func = $table["Function"]
    $match = $table["Match"]
    $version = $table["Version"]
    $noInstallRequired = $true

    $allProducts = LoadWin32Product
    $productList = @($allProducts | Where-Object { $_.Name -match $match } )

    if ($productList.Count -eq 0) {
        Write-Verbose "No product found with Name matching [$match]"
        $noInstallRequired = $false
    }
    else {
        $productList = @($productList | Where-Object { $_.Version -lt $version })
        if ($productList.Count -gt 0) {
            Write-Verbose "Products with earlier versions found`n$productList"
            $noInstallRequired = $false
        }
    }

    Write-Verbose "[$func]: Product [$match] Version {$version] returned [$noInstallRequired]"
    return $noInstallRequired
}

function VerifyInstallationContent(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func = $table["Function"]
    $path = $table["Path"]

    $noInstallRequired = (join-path $path cntk\cntk.exe | test-path -PathType Leaf) 
    $noInstallRequired = (join-path $path prerequisites\VS2015\vc_redist.x64.exe | test-path -PathType Leaf) -and $noInstallRequired
    $noInstallRequired = (join-path $path prerequisites\MSMpiSetup.exe | test-path -PathType Leaf) -and $noInstallRequired

    if ($noInstallRequired) {
        Write-Verbose "[$func]: [$path] returned [$noInstallRequired]"
        return $noInstallRequired
    }
    
    throw "`nFatal Error: Files from the CNTK binary download package are missing!`nThe install script must be run out of the unpacked binary CNTK package, not from a CNTK source clone."
}

function VerifyDirectory(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func = $table["Function"]
    $path = $table["Path"]

    $noInstallRequired = (test-path -path $path -PathType Container)

    Write-Verbose "[$func]: [$path] returned [$noInstallRequired]"
    
    return $noInstallRequired
}

function VerifyRunAlways(
	[Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table
    $func = $table["Function"]

    $noInstallRequired = $false
    Write-Verbose "[$func]: returned [$noInstallRequired]"
    return $noInstallRequired
}

function VerifyFile(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func = $table["Function"]
    $path = $table["Path"]

    $noInstallRequired = (test-path -path $path -PathType Leaf)

    Write-Verbose "[$func]: [$path] returned [$noInstallRequired]"
    
    return $noInstallRequired
}

function VerifyRegistryKey(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func = $table["Function"]
    $key = $table["Key"]

    $noInstallRequired = (test-path -path $key)

    Write-Verbose "[$func]: [$key] returned [$noInstallRequired]"
    
    return $noInstallRequired
}

function VerifyRegistryKeyName(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func     = $table["Function"]
    $key      = $table["Key"]
    $regName  = $table["RegName"]

    $noInstallRequired = Test-ItemProperty -Path $key -Name $regName

    Write-Verbose "[$func]: [$key]:[$regname] returned [$noInstallRequired]"
    
    return $noInstallRequired
}

function VerifyRegistryKeyNameData(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func     = $table["Function"]
    $key      = $table["Key"]
    $regName  = $table["RegName"]
    $regData  = $table["RegData"]

    $noInstallRequired = (test-path -path $key)

    if ($noInstallRequired) {
        $theKeyObj = get-item $key
        $noInstallRequired = ($theKeyObj.GetValue("$regName") -eq $regData)
    }

    Write-Verbose "[$func]: [$key]:[$regname] == [$regData] returned [$noInstallRequired]"
    return $noInstallRequired
}

function Test-ItemProperty (
    [string] $Path, 
    [string] $Name)
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

function LoadWin32Product
{
    if ($Script:Win32Product -eq $Null) {
        $Script:Win32Product = Get-WmiObject Win32_Product 
    }

    return $Script:Win32Product
}
