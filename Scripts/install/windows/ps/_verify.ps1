#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#
function VerifyOperations(
    [hashtable[]] $operations,
    [bool] $NoConfirm)
{
    $operationList  = @()

    Write-Host "Determining Operations to perform. This will take a moment..."

    foreach ($item in $operations) {
        $needsInstall = $false

        foreach ($verificationItem in $item.Verification) {
            $params = $verificationItem.Params
            $expr = $verificationItem.Function +' @params' 
        
            Write-Verbose "Calling Operation: [$expr]($params)"
            $needsInstall = Invoke-Expression $expr
            
            if (-not $needsInstall) {
                $operationList += $item
                break
            }
        }
    }

    Write-Host 

    if ($operationList) {
        Write-Host "The following operations will be performed:"

        foreach ($item in $operationList) {
            $info = $item.Info
            Write-Host " * $info"
        }
        if ($NoConfirm) {
            return $operationList
        }
        Write-Host 
        Write-Host "Do you want to continue? (y/n)"
        
        $choice = GetKey '^[yYnN]+$'

        if ($choice -contains "y") {
            return $operationList
        }
    }
    else {
        Write-Host "No additional installation required"
    }
    return @()
}

function VerifyScanPrograms
{
    $noInstallRequired = $true
    
    # no actual work is being performed, just the script local datastructure with the list
    # of installed programs is being initialized
    LoadWinProduct
    return $noInstallRequired
}

function VerifyWinProductExists(
    [Parameter(Mandatory = $true)][string] $match)
{
    $noInstallRequired = $true

    $allProducts = LoadWinProduct
    $productList = @($allProducts | Where-Object { $_.Name -match $match } )
    
    if ($productList.Count -eq 0) {
        $noInstallRequired = $false
    }

    Write-Verbose "[$MyInvocation.MyCommand]: Product [$match] returned [$noInstallRequired]"
    return $noInstallRequired
}

function VerifyWinProductVersion(
    [Parameter(Mandatory = $true)][string] $match,
    [Parameter(Mandatory = $true)][string] $version)
{
    $noInstallRequired = $true

    $allProducts = LoadWinProduct
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

    Write-Verbose "[$MyInvocation.MyCommand]: Product [$match] Version [$version] returned [$noInstallRequired]"
    return $noInstallRequired
}

function VerifyDirectory(
    [Parameter(Mandatory = $true)][string] $path)
{
    $noInstallRequired = (test-path -path $path -PathType Container)

    Write-Verbose "[$MyInvocation.MyCommand]: [$path] returned [$noInstallRequired]"
    
    return $noInstallRequired
}

function VerifyRunAlways
{
    $noInstallRequired = $false
    Write-Verbose "[$MyInvocation.MyCommand]: returned [$noInstallRequired]"
    return $noInstallRequired
}

function VerifyFile(
    [Parameter(Mandatory = $true)][string] $longFileName)
{
    $noInstallRequired = (Test-Path -path $longFileName -PathType Leaf)

    Write-Verbose "[$MyInvocation.MyCommand]: [$longFileName] returned [$noInstallRequired]"
    
    return $noInstallRequired
}

function LoadWinProduct
{
    if (-not $Script:WinProduct) {
        # 
        # $Script:WinProduct = Get-WmiObject Win32_Product
        # The above line was the previous solution, but iterating through the registry is much faster
        # get-wmiobject does more house-holding, like checking for consistency etc ...
        # 
        $allInstalled = @(Get-ChildItem "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall" -ErrorAction SilentlyContinue) + 
                        @(Get-ChildItem "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall" -ErrorAction SilentlyContinue) + 
                        @(get-ChildItem "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\" -ErrorAction SilentlyContinue)

        $result = @()
        foreach ($item in $allInstalled) {
            $displayName = $item.GetValue("DisplayName")
            if ($displayName) {
                $entry = New-Object PSObject
                $entry | Add-Member -MemberType NoteProperty -Name "Name" -Value $displayName
                $entry | Add-Member -MemberType NoteProperty -Name "Version" -Value $($item.GetValue("DisplayVersion"))
                
                $result += $entry
            }

        } 
        $result = $result | Sort-Object Name,Version -Unique

        $Script:WinProduct = $result
    }
    return $Script:WinProduct
}

# vim:set expandtab shiftwidth=4 tabstop=4: