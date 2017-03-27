#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#
function GetTableDefaultBool(
    [hashtable] $table,
    [string] $entryName,
    [bool] $defaultValue
)
{
    if ($table[$entryName] -eq $null) {
        return $defaultValue
    }
    return $table[$entryName]
}

function GetTableDefaultString(
    [hashtable] $table,
    [string] $entryName,
    [string] $defaultValue
)
{
    if ($table[$entryName] -eq $null) {
        return $defaultValue
    }
    return $table[$entryName]
}

function GetTableDefaultInt(
    [hashtable] $table,
    [string] $entryName,
    [int] $defaultValue
)
{
    if ($table[$entryName] -eq $null) {
        return $defaultValue
    }
    return $table[$entryName]
}

function GetEnvironmentVariableContent(
    [string] $envVarName)
{
    $envContent = [environment]::GetEnvironmentVariable($envVarName, "USER")
    if ($envContent) {
        return $envContent
    }
    $envContent = [environment]::GetEnvironmentVariable($envVarName, "MACHINE")
    return $envContent
}

# vim:set expandtab shiftwidth=4 tabstop=4: