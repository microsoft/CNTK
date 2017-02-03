#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#
function VerifyOperations(
    [array] $verificationList)
{
    Write-Host "Determining Operations to perform. This will take a moment..."

    $result = @()

    foreach ($item in $verificationList) {
        $needsInstall = $false
        Write-Host $item.VerifyInfo
        foreach ($verificationItem in $item.Verification) {
            $needsInstall = VerifyItem $verificationItem
            if (-not $needsInstall) {
                $result += $item
                break
            }
        }
    }
    return $result
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
    LoadWinProduct
    return $noInstallRequired
}

function VerifyWinProductExists(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table
    $func = $table["Function"]
    $match = $table["Match"]
    $compare = GetTableDefaultString -table $table -entryName "Compare" -defaultValue ""
    $matchExact = GetTableDefaultBool -table $table -entryName "MatchExact" -defaultValue $true
    $noInstallRequired = $true

    $allProducts = LoadWinProduct
    $productList = @($allProducts | Where-Object { $_.Name -match $match } )

    if ($productList.Count -eq 0) {
        $noInstallRequired = $false
    }
    elseif ($compare.length -gt 0) {
        if ($matchExact) {
            $productList = @($productList | Where-Object { $_.Name -eq $compare })
        }
        else {
            $productList = @($productList | Where-Object { $_.Version -ge $compare })
        }
        if ($productList.Count -eq 0) {
            Write-Verbose "No product found matching the compare requirement`n$productList"
            $noInstallRequired = $false
        }
    }

    Write-Verbose "[$func]: Product [$match] returned [$noInstallRequired]"
    return $noInstallRequired
}

function VerifyWinProductVersion(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table
    $func = $table["Function"]
    $match = $table["Match"]
    $version = $table["Version"]
    $matchExact = GetTableDefaultBool -table $table -entryName "MatchExact" -defaultValue $false
    $noInstallRequired = $true

    $allProducts = LoadWinProduct
    $productList = @($allProducts | Where-Object { $_.Name -match $match } )

    if ($productList.Count -eq 0) {
        Write-Verbose "No product found with Name matching [$match]"
        $noInstallRequired = $false
    }
    else {
        if ($matchExact) {
            $productList = @($productList | Where-Object { $_.Version -eq $version })
        }
        else {
            $productList = @($productList | Where-Object { $_.Version -ge $version })
        }
        if ($productList.Count -eq 0) {
            Write-Verbose "No product found matching the version requirement`n$productList"
            $noInstallRequired = $false
        }
    }
    Write-Verbose "[$func]: Product [$match] Version {$version] returned [$noInstallRequired]"
    return $noInstallRequired
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

function VerifyWheelDirectory(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table
    $func = $table["Function"]
    $path = $table["WheelDirectory"]
    $forceUpdate = $table["ForceUpdate"]

    $noInstallRequired = $false

    Write-Verbose "[$func]: [$path] returned [$noInstallRequired]"
    return $noInstallRequired
}

function VerifyPathIncludes(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table
    $func = $table["Function"]
    $path = $table["Path"]

    $noInstallRequired = (test-path -path $path -PathType Container)

    Write-Verbose "[$func]: [$path] returned [$noInstallRequired]"
    return $noInstallRequired
}

function VerifyDirectoryContent(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table
    $func = $table["Function"]
    $source = $table["Source"]
    $dest = $table["Destination"]

    $noInstallRequired = (test-path -path $source -PathType Container)

    if ($noInstallRequired) {
        $noInstallRequired = (test-path -path $dest -PathType Container)
    }
    if ($noInstallRequired) {
        $r = Compare-Object $(Get-ChildItem $source -Recurse) $(Get-ChildItem $dest -Recurse)
        if ($r) {
            $noInstallRequired = $false
        }
    }

    Write-Verbose "[$func]: [$source] with [$dest] returned [$noInstallRequired]"
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
    $orLater  = $table["OrLater"]

    if ($orLater -eq $null) {
        $orLater = $false
    }

    $noInstallRequired = (test-path -path $key)
    if ($noInstallRequired) {
        $theKeyObj = get-item $key
        $theKeyValue = $theKeyObj.GetValue("$regName")
        $noInstallRequired = $false

        if ($theKeyValue -ne $null) {
            if ($orLater) {
                $noInstallRequired = ($theKeyValue -ge $regData)
            }
            else {
                $noInstallRequired = ($theKeyValue -eq $regData)
            }
        }
    }

    Write-Verbose "[$func]: [$key]:[$regname] == [$regData] returned [$noInstallRequired]"
    return $noInstallRequired
}

function VerifyEnvironmentAndData(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table
    $func    = $table["Function"]
    $name    = $table["EnvVar"]
    $content = $table["Content"]
    $location = "User"

    $envContent = GetEnvironmentVariableContent $name 
    $noInstallRequired = $envContent -eq $content

    Write-Verbose "[$func]: [$name] == [$content] returned [$noInstallRequired]"
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

function ClearScriptVariables
{
    $Script:WinProduct = $Null
}

function LoadWinProduct
{
    if ($Script:WinProduct -eq $Null) {
        # 
        # $Script:WinProduct = Get-WmiObject Win32_Product
        # The above line was the previous solution, but iterating through the registry is much faster
        # get-wmiobject does more house-holding, like checking for concistency etc ...
        # 
        $allInstalled = @(Get-childitem "HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall" -ErrorAction SilentlyContinue) + `
                       @(Get-childitem "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall" -ErrorAction SilentlyContinue) + `
                       @(get-childitem "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\" -ErrorAction SilentlyContinue)

        $result = @()
        foreach($item in $allInstalled) {
            $displayName = $item.GetValue("DisplayName")
            if ($displayName.Length -gt 0) {
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