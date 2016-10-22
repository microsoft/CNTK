function DownloadOperations()
{
    Write-Host "Performing download operations"

    foreach ($item in $global:operationList) {
        
        foreach ($downloadItem in $item.Download) {
            DownloadItem $downloadItem
        }
    }

    Write-Host "Download operations finished"
    Write-Host
}


function DownloadItem(
    [hashtable] $item
){
    $func = $item["Function"]

    $expr = $func +' $item' 
        
    Write-Verbose "Calling Operation: [$func]"
    $result = Invoke-Expression $expr 

    return
}


function Download(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table

    $func = $table["Function"]
    $source = $table["Source"]
    $destination = $table["Destination"]

    $result = DownloadFile -SourceFile $source -OutFile $destination

    Write-Verbose "[$func]: [$source -> $destination] returned [$result]"
    
    return $result
}

function DownloadAndExtract(
    [string] $tPath,
    [string] $sAddress,
    [string] $fileName,
    [string] $targetPathRoot
){
    $outFileName  = Join-Path $tPath $fileName

    DownloadFile -SourceFile $sAddress `
                    -OutFile $outFileName `
                    -tempFileName $fileName

    Write-Host Extracting into targetpath
    Write-Host

    ExtractAllFromZip $outFileName $targetPathRoot
}

function DownloadFile (
    [string] $SourceFile,
    [string] $OutFile,
    [int] $timeout = 600,
    [int] $maxtry = 5
)
{
    # the scriptblock to invoke a web-request
    $sb ={
            param([string]$uri,[string]$outfile)
            (New-Object System.Net.WebClient).DownloadFile($uri,$outfile) 
         }
    #---------------

    $startTime = Get-Date
    Write-Host "Downloading [$SourceFile], please be patient...."
    if (-not $Execute) {
         Write-Host  "$message ** Running in DEMOMODE - no download performed"
         return $true
    }

    $TempFile = [System.IO.Path]::GetTempFileName()

    for ($count=1; $count -le $maxtry; $count +=1) {
        if ($count -gt 1) {
            Write-Host "Iteration [$count] of [$maxtry]"
        }
        
        if ($count -gt 1) {
            start-sleep -Seconds 5
        }
        if (Test-Path -Path $TempFile) {
            # the file we temporary use as a download cache could exist
            # if it does, we remove it and terminate if this fails
            Remove-Item -path $TempFile -ErrorAction Stop
        }    
        
        if (test-path -path $outFile) {
            Write-Host "File [$outFile] already exists"
            return $true
        }   

        $job = start-job -scriptblock $sb -ArgumentList $sourceFile, $TempFile
        Wait-Job $job -Timeout $timeout

        $jState = $job.State.ToUpper()
        $jStart = $job.PSBeginTime
        $jEnd = $job.PSEndTime
        $jError =  $job.ChildJobs[0].Error
        $current = Get-Date

        switch ($jState) {
            "COMPLETED" { 
                            if ($jError.Count -eq 0) {
                                Write-Host "End binary download!"

                                Remove-Job $job -force -ErrorAction SilentlyContinue

                                # we now have the temporary file, we need to rename it
                                move-Item $TempFile $OutFile -ErrorAction SilentlyContinue

                                if (Test-Path -Path $OutFile) {
                                    # we have a file with our expected filename, we are in good shape and ready to report success
                                    # in case the above rename failed, but the target file exist, we clean up a possible dangling TempFile
                                    # no need to check if this file is really there. we don't care if it succeeds or not
                                    Remove-Item -path $TempFile -ErrorAction SilentlyContinue

                                    return $true
                                }

                                # we got here because we finished the job, but some operation failed (i.e. the rename above. we can just try again)
                                Write-Verbose "Job completed but rename operation failed, retrying..."
                                continue
                            }

                            Write-Host "Job Completed with Error: [$jStart] to [$current]"
                            Write-Host $jError
                        }
            "RUNNING"   {
                            Write-Host "Job Timeout: [$jStart] to [$current]"
                        }
            "FAILED"    {
                            $current = Get-Date
                            Write-Host "Job Failed: [$jStart] to [$current]"
                            Write-Host "Error: $jError"
                        }
            default     {
                            Write-Host "Job State: [$Error] - [$jStart] to [$current]"
                        }
        }
        Remove-Job $job -force -ErrorAction SilentlyContinue
    }

    throw "Download $SourceFile Failed!"
}

