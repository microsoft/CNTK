function OpAnaconda3411(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "Anaconda3-4.1.1"
    $prodFile = "Anaconda3-4.1.1-Windows-x86_64.exe"
    $prodSubDir = "Anaconda3-4.1.1-Windows-x86_64" 
    $targetPath = join-path $targetFolder $prodSubDir
    $downloadSource = "https://repo.continuum.io/archive/Anaconda3-4.1.1-Windows-x86_64.exe"
    $downloadSize = 370055720

    @( @{ShortName = "ANA3-411"; Name = $prodName;  VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath; } );
         Download = @( @{Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedSize = $downloadSize } );
         Action = @( @{Function = "InstallExe"; Command = "$cache\$prodFile"; Param = "/InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=$targetPath"; runAs=$false; Message = ".... This will take some time. Please be patient ...." } );
     } )
}

function OpAnacondaEnv34(
    [parameter(Mandatory=$true)][string] $targetFolder,
    [parameter(Mandatory=$true)][string] $repoDir,
    [parameter(Mandatory=$true)][string] $reponame)
{
    $prodName = "Python 3.4 Environment"

    $prodSubDir = "Anaconda3-4.1.1-Windows-x86_64" 
    $targetPath = join-path $targetFolder $prodSubDir
    $envName = "cntkdev-py34"
    $envDir = "envs\$envName"
    $envVar = "CNTK_PY34_PATH";
    $envValue = join-path $targetPath $envDir

    $ymlDirectory = join-path $repoDir $repoName
    $ymlDirectory = join-path $ymlDirectory "scripts\install\windows"

    @{ ShortName = "PYENV34"; Name = $prodName;  VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Creating $prodName";
      Verification  = @( @{Function = "VerifyRunAlways" } );
      Action = @( @{Function = "InstallYml"; BasePath = $targetPath; Env = $envName; ymlFile= "$ymlDirectory\conda-windows-cntk-py34-environment.yml" },
                  @{Function = "SetEnvironmentVariable"; EnvVar= $envVar; Content = $envValue } )
     }
}

function OpBoost160(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "Boost 1.60.0"
    $prodFile = "boost_1_60_0-msvc-12.0-64.exe"
    $prodSubDir = "boost_1_60_0"
    $targetPath = join-path $targetFolder $prodSubDir
    $downloadSource = "https://sourceforge.net/projects/boost/files/boost-binaries/1.60.0/boost_1_60_0-msvc-12.0-64.exe/download"

    @( @{Name = $prodName; ShortName = "BOOST160"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath },
                        @{Function = "VerifyEnvironmentAndData"; EnvVar = "BOOST_INCLUDE_PATH"; Content = $targetPath },
                        @{Function = "VerifyEnvironmentAndData"; EnvVar = "BOOST_LIB_PATH"; Content  = "$targetPath\lib64-msvc-12.0" } );
        Download = @( @{Function = "Download"; Source = $downloadSource; Destination = "$cache\$prodFile" } );
        Action = @( @{Function = "SetEnvironmentVariable"; EnvVar = "BOOST_INCLUDE_PATH"; Content = "$targetPath" },
                    @{Function = "SetEnvironmentVariable"; EnvVar = "BOOST_LIB_PATH"; Content = "$targetPath\lib64-msvc-12.0" },
                    @{Function = "InstallExe"; Command = "$cache\$prodFile"; Param = "/dir=$targetPath /SP- /SILENT /NORESTART"; runAs=$false } );
     } )
}

function OpBoost160VS15(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "Boost 1.60.0"
    $prodFile = "boost_1_60_0-msvc-14.0-64.exe"
    $prodSubDir = "boost_1_60_0-msvc-14.0"
    $targetPath = join-path $targetFolder $prodSubDir
    $downloadSource = "https://sourceforge.net/projects/boost/files/boost-binaries/1.60.0/boost_1_60_0-msvc-14.0-64.exe/download"
    $envVar = "BOOST_INCLUDE_PATH"
    $envVarLib = "BOOST_LIB_PATH"
    $envContentLib = "$targetPath\lib64-msvc-14.0"

    @( @{Name = $prodName; ShortName = "BOOST160VS15"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath },
                        @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $targetPath },
                        @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVarLib; Content  = $envContentLib } );
        Download = @( @{Function = "Download"; Source = $downloadSource; Destination = "$cache\$prodFile" } );
        Action = @( @{Function = "SetEnvironmentVariable"; EnvVar = $envVar; Content = $targetPath },
                    @{Function = "SetEnvironmentVariable"; EnvVar = $envVarLib; Content  = $envContentLib },
                    @{Function = "InstallExe"; Command = "$cache\$prodFile"; Param = "/dir=$targetPath /SP- /SILENT /NORESTART"; runAs=$false } );
     } )
}

function OpCMake362(
    [parameter(Mandatory=$true)][string] $cache)
{
    $prodName = "CMake 3.6.2"
    $targetPath = join-path $env:ProgramFiles "cmake\bin"
    $cmakeName = "cmake-3.6.2-win64-x64.msi"
    $downloadSource = "https://cmake.org/files/v3.6/cmake-3.6.2-win64-x64.msi"
    
    @( @{ShortName = "CMake362"; Name = $prodName; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
        Verification = @( @{Function = "VerifyWinProductExists"; Match = "^CMake$"; Version = "3.6.2" }  );
        Download = @( @{Function = "Download"; Source = $downloadSource; Destination = "$cache\$cmakeName" } );
        Action = @( @{Function = "InstallMsi"; MsiName =  "$cmakeName" ; MsiDir   = "$cache" }  ,
                    @{Function = "AddToPath"; Dir = "$targetPath" } );
        } )
}

function OpCNTKMKL2
    ([parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "CNTK Custom MKL Version 2"
    $prodFile = "CNTKCustomMKL-Windows-2.zip"
    $prodSubDir = "CNTKCustomMKL"
    $targetPath = join-path $targetFolder $prodSubDir
    $targetPathCurrenVersion = join-path $targetPath "2"
    $envVar = "CNTK_MKL_PATH";
    $envValue = $targetPath
    $downloadSource = "https://www.cntk.ai/mkl/$prodFile";

    @(  @{ShortName = "CNTKMKL2"; Name = $prodName; VerifyInfo = "Checking for $prodName in $targetPathCurrenVersion"; ActionInfo = "Installing $prodName"; 
          Verification = @( @{Function = "VerifyDirectory"; Path = $targetPathCurrenVersion },
                            @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
          Download = @( @{ Function = "Download"; Source = $downloadSource; Destination = "$cache\$prodFile" } );
          Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $localDir; destinationFolder = $prodSubDir },
                      @{Function = "SetEnvironmentVariable"; EnvVar= $envVar; Content = $envValue } );
         } )
}

function OpCNTKMKL3
    ([parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "CNTK Custom MKL Version 3"
    $prodFile = "CNTKCustomMKL-Windows-3.zip"
    $prodSubDir = "CNTKCustomMKL"
    $targetPath = join-path $targetFolder $prodSubDir
    $targetPathCurrenVersion = join-path $targetPath "3"
    $envVar = "CNTK_MKL_PATH";
    $envValue = $targetPath
    $downloadSource = "https://www.cntk.ai/mkl/$prodFile";

    @(  @{ShortName = "CNTKMKL3"; Name = $prodName; VerifyInfo = "Checking for $prodName in $targetPathCurrenVersion"; ActionInfo = "Installing $prodName"; 
          Verification = @( @{Function = "VerifyDirectory"; Path = $targetPathCurrenVersion },
                            @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
          Download = @( @{ Function = "Download"; Source = $downloadSource; Destination = "$cache\$prodFile" } );
          Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $localDir; destinationFolder = $prodSubDir },
                      @{Function = "SetEnvironmentVariable"; EnvVar= $envVar; Content = $envValue } );
         } )
}

function OpCygwin
    ([parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "CygWin64"
    $prodFile = "cygwinSetup_64.exe"
    $prodSubDir = "cygwin64"
    $targetPath = join-path $targetFolder $prodSubDir
    $regkey = "Registry::HKEY_CURRENT_USER\SOFTWARE\Cygwin"
    $downloadSource = "https://www.cygwin.com/setup-x86_64.exe"
    $installParam =  "--quiet-mode --no-admin --site http://mirrors.kernel.org/sourceware/cygwin/ --root $targetPath --local-package-dir $cache  --packages python,python-yaml,python-numpy,diffutils"
    $bFileName = "cygwinpip.bash"
    $bashFileName = join-path $targetFolder $bFileName
    $bashParmFile = Join-Path "\cygdrive\c" (split-path $bashFileName -NoQualifier)
    $bashParmFile = $bashParmFile.replace("\","/")
    $bashParam = "-l -c $bashParmFile"

    @(  @{ShortName = "CYGWIN64"; Name = $prodName; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName"; 
          Verification = @( @{Function = "VerifyDirectory"; Path = "$targetPath" },
                            @{Function = "VerifyRegistryKey"; Key = $regKey } );
          Download = @( @{ Function = "Download"; Source = $downloadSource; Destination = "$cache\$prodFile" } );
          Action = @( @{Function = "InstallExe"; Command = "$cache\$prodFile"; Param = $installParam; runAs=$false; WorkDir = $targetFolder},
                      @{Function = "AddToPath"; Dir = "$targetPath\bin"; AtStart  = $false },
                      @{Function = "CreateCygwinBashScript"; FileName = $bashFileName },
                      @{Function = "InstallExe"; Command = "$targetPath\bin\bash.exe"; Param = $bashParam; runAs=$false; WorkDir = $targetPath } );
         } )
}

function AddOpDisableJITDebug
{
    @(   @{Name="Visual Studio Disable Just-In-Time Debugging"; ShortName = "JITVS2013"; VerifyInfo = "Checking for JIT Debugging registry keys"; ActionInfo = "Removing registry keys to disable JIT debugging, advisable for Jenkins machines";
          Verification = @( @{Function = "DeletedRegistryKeyName"; Key = "Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion\AeDebug"; RegName  = "Debugger"; },
                            @{Function = "DeletedRegistryKeyName"; Key = "Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\.NETFramework"; RegName  = "DbgManagedDebugger" },
                            @{Function = "DeletedRegistryKeyName"; Key = "Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Microsoft\Windows NT\CurrentVersion\AeDebug"; RegName  = "Debugger" }, 
                            @{Function = "DeletedRegistryKeyName"; Key = "Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Microsoft\.NETFramework"; RegName = "DbgManagedDebugger" } );
          Action = @( @{Function = "RemoveRegistryKeyName"; Key = "Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion\AeDebug"; Elevated = $true; RegName = "Debugger" },
                      @{Function = "RemoveRegistryKeyName"; Key = "Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\.NETFramework"; Elevated = $true; RegName = "DbgManagedDebugger" },
                      @{Function = "RemoveRegistryKeyName"; Key = "Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Microsoft\Windows NT\CurrentVersion\AeDebug"; Elevated = $true; RegName = "Debugger" }, 
                      @{Function = "RemoveRegistryKeyName"; Key = "Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Microsoft\.NETFramework"; Elevated = $true; RegName  = "DbgManagedDebugger" } )
          } )
 }

function OpGit2101(
    [parameter(Mandatory=$true)][string] $cache)
{
    $prodName = "Git"
    $targetPath = join-path $env:ProgramFiles "Git\bin"
    $prodFile = "Git-2.10.1-64-bit.exe"
    $downloadSource = "https://github.com/git-for-windows/git/releases/download/v2.10.1.windows.1/$prodFile"

    @( @{ShortName = "GIT"; Name = $prodName; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
         Verification  = @( @{Function = "VerifyRegistryKeyName"; Key = "Registry::HKEY_LOCAL_MACHINE\SOFTWARE\GitForWindows"; RegName = "CurrentVersion"; } );
         Download = @( @{ Function = "Download"; Source = $downloadSource; Destination = "$cache\$prodFile" } );
         Action = @( @{Function = "InstallExe"; Command = "$cache\$prodFile"; Param = "/SP- /SILENT /NORESTART"},
                     @{Function = "AddToPath"; Dir = $targetPath; AtStart  = $true; } )
     } )
}

function OpGitClone(
    [parameter(Mandatory=$true)][string] $targetFolder,
    [parameter(Mandatory=$true)][string] $targetDir,
    [string] $repoTag = "master")
{
    $targetPath = join-path $targetFolder $targetDir
    $downloadSource = "https://github.com/Microsoft/CNTK/";
    $appDir = join-path $env:ProgramFiles "Git\bin"

   @( @{Name = "Clone CNTK from Github"; ShortName = "CNTKCLONE"; VerifyInfo = "Checking for CNTK-Clone target directory $targetPath"; ActionInfo = "Cloneing CNTK from Github repository";
      Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath } );
      Action = @( @{Function = "MakeDirectory"; Path = $targetFolder },
                  @{Function = "ExecuteApplication"; AppName = "git.exe"; Param = "clone --branch $repoTag --recursive https://github.com/Microsoft/CNTK/"; AppDir = $appDir; UseEnvPath = $true; WorkDir = $targetFolder } )
     } )
}

function OpMSMPI70([parameter(
    Mandatory=$true)][string] $cache)
{
    $remoteFilename = "MSMpiSetup.exe"
    $localFilename = "MSMpiSetup70.exe"
    $downloadSource = "https://download.microsoft.com/download/D/7/B/D7BBA00F-71B7-436B-80BC-4D22F2EE9862/$remoteFilename";

    @( @{Name = "MSMPI Installation"; ShortName = "CNTK"; VerifyInfo = "Checking for installed MSMPI 70"; ActionInfo = "Installing MSMPI 70";
         Verification = @( @{Function = "VerifyWinProductVersion"; Match = "^Microsoft MPI \(\d+\."; Version = "7.0.12437.6"; MatchExact = $false } );
         Download = @( @{Function = "Download"; Source = $downloadSource; Destination = "$cache\$localFilename" } );
         Action = @( @{Function = "InstallExe"; Command =  "$cache\$localFilename" ; Param = "/unattend" } )
        } )
}

function OpMSMPI70SDK(
    [parameter(Mandatory=$true)][string] $cache)
{
    $remoteFilename = "msmpisdk.msi"
    $localFilename = "msmpisdk70.msi"
    $downloadSource = "https://download.microsoft.com/download/D/7/B/D7BBA00F-71B7-436B-80BC-4D22F2EE9862/$remoteFilename";

    @( @{Name = "MSMPI SDK70 Installation"; ShortName = "CNTK"; VerifyInfo = "Checking for installed MSMPI 70 SDK"; ActionInfo = "Install MSMPI 70 SDK";
         Verification = @( @{Function = "VerifyWinProductVersion"; Match = "^Microsoft MPI SDK \(\d+\."; Version = "7.0.12437.6"; MatchExact = $false } );
         #Verification = @( @{Function = "VerifyWinProductExists"; Match = "^Microsoft MPI SDK \(\d+\."; Compare = "^Microsoft MPI SDK \(7\.0\.12437\.6\)"; MatchExact = $false } );
         Download = @( @{Function = "Download"; Source = $downloadSource; Destination = "$cache\$localFilename" } );
         Action = @( @{Function = "InstallMsi"; MsiName =  "$localFilename" ; MsiDir   = "$cache" } )
        } )
}

function OpNvidiaCub141(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "NVidia CUB 1.4.1"
    $prodFile = "cub-1.4.1.zip"
    $prodSubDir = "cub-1.4.1"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "CUB_PATH";
    $envValue = $targetPath
    $downloadSource = "https://codeload.github.com/NVlabs/cub/zip/1.4.1";

    @( @{ShortName = "CUB141"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
         Verification = @( @{Function = "VerifyDirectory"; Path = "$targetPath" },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
         Download = @( @{Function = "Download"; Source = $downloadSource; Destination = "$cache\$prodFile" } );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = "$targetFolder"; destinationFolder = $prodSubDir; zipSubTree= $prodSubDir },
                     @{Function = "SetEnvironmentVariable"; EnvVar= $envVar; Content = $envValue } );
         } )
}

function OpNVidiaCuda75(
    [parameter(Mandatory=$true)][string] $cache)
{
    $cudaDownload = "http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers"
    $programPath = join-path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA\v7.5"

    
    @( @{Name="NVidia CUDA 7.5"; ShortName = "CUDA75"; VerifyInfo = "Checking for installed NVidia Cuda 75"; ActionInfo = "Installing CUDA 7.5";
         Verification = @( @{Function = "VerifyDirectory"; Path = $programPath },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = "CUDA_PATH_V7_5"; Content = $programPath } );
         Download = @( @{Function = "DownloadForPLatform"; Platform = "^Microsoft Windows (7|8|Server 2008 R2|Server 2012 R2)"; Source = "$cudaDownload/cuda_7.5.18_windows.exe"; Destination = "$cache\cuda_7.5.18_windows.exe" },
                       @{Function = "DownloadForPLatform"; Platform = "^Microsoft Windows 10"; Source = "$cudaDownload/cuda_7.5.18_win10.exe"; Destination = "$cache\cuda_7.5.18_win10.exe" } );
         Action = @( @{Function = "InstallExeForPlatform"; Platform = "^Microsoft Windows (7|8|Server 2008 R2|Server 2012 R2)"; Command = "$cache\cuda_7.5.18_windows.exe"; Param = "-s CUDAToolkit_7.5 CUDAVisualStudioIntegration_7.5 GDK"; Message = ".... This will take some time. Please be patient ...." } ,
                     @{Function = "InstallExeForPlatform"; Platform = "^Microsoft Windows 10"; Command = "$cache\cuda_7.5.18_win10.exe"; Param = "-s CUDAToolkit_7.5 CUDAVisualStudioIntegration_7.5 GDK"; Message = ".... This will take some time. Please be patient ...." }  );
         })
}

function OpNVidiaCuda80(
    [parameter(Mandatory=$true)][string] $cache)
{
    $cudaDownload = "http://developer.download.nvidia.com/compute/cuda/8.0/secure/prod/local_installers"
    $cudaFile = "cuda_8.0.44_windows.exe"
    $cudaFileWin10 = "cuda_8.0.44_win10.exe"
    $programPath = join-path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA\v8.0"
    $cudaInstallParam = "-s"

    @( @{Name="NVidia CUDA 8.0"; ShortName = "CUDA80"; VerifyInfo = "Checking for installed NVidia Cuda 80"; ActionInfo = "Installing CUDA 8.0";
         Verification = @( @{Function = "VerifyDirectory"; Path = $programPath },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = "CUDA_PATH_V8_0"; Content = $programPath } );
         Download = @( @{Function = "DownloadForPLatform"; Platform = "^Microsoft Windows (7|8|Server 2008 R2|Server 2012 R2)"; Source = "$cudaDownload/$cudaFile"; Destination = "$cache\$cudaFile" },
                       @{Function = "DownloadForPLatform"; Platform = "^Microsoft Windows 10"; Source = "$cudaDownload/$cudaFileWin10"; Destination = "$cache\$cudaFileWin10" } );
         Action = @( @{Function = "InstallExeForPlatform"; Platform = "^Microsoft Windows (7|8|Server 2008 R2|Server 2012 R2)"; Command = "$cache\$cudaFile"; Param = $cudaInstallParam } ,
                     @{Function = "InstallExeForPlatform"; Platform = "^Microsoft Windows 10"; Command = "$cache\$cudaFileWin10"; Param = $cudaInstallParam }  );
         })
}

function OpNVidiaCudnn5175(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)

{
    $prodName = "NVidia CUDNN 5.1 for CUDA 7.5"
    $cudnnWin7 = "cudnn-7.5-windows7-x64-v5.1.zip"
    $cudnnWin10 = "cudnn-7.5-windows10-x64-v5.1.zip"

    $prodSubDir =  "cudnn-7.5-v5.1"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "CUDNN_PATH"
    $envValue = join-path $targetPath "cuda"
    $downloadSource = "http://developer.download.nvidia.com/compute/redist/cudnn/v5.1"

    @( @{ShortName = "CUDNN5175"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
         Download = @( @{Function = "DownloadForPLatform"; Platform = "^Microsoft Windows 7"; Source = "$downloadSource/$cudnnWin7"; Destination = "$cache\$cudnnWin7" },
                       @{Function = "DownloadForPLatform"; Platform = "^Microsoft Windows (8|10|Server 2008 R2|Server 2012 R2)"; Source = "$downloadSource/$cudnnWin10"; Destination = "$cache\$cudnnWin10" } );
         Action = @( @{Function = "ExtractAllFromZipForPlatform"; Platform = "^Microsoft Windows 7"; zipFileName = "$cache\$cudnnWin10"; destination = $targetFolder; destinationFolder = $prodSubDir },
                     @{Function = "ExtractAllFromZipForPlatform"; Platform = "^Microsoft Windows (8|10|Server 2008 R2|Server 2012 R2)"; zipFileName = "$cache\$cudnnWin10"; destination = $targetFolder; destinationFolder = $prodSubDir },
                     @{Function = "SetEnvironmentVariable"; EnvVar = $envVar; Content  = $envValue } );
         })
}

function OpNVidiaCudnn5180(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "NVidia CUDNN 5.1 for CUDA 8.0"
    $cudnnWin7 = "cudnn-8.0-windows7-x64-v5.1.zip"
    $cudnnWin10 = "cudnn-8.0-windows10-x64-v5.1.zip"

    $prodSubDir =  "cudnn-8.0-v5.1"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "CUDNN_PATH"
    $envValue = join-path $targetPath "cuda"
    $downloadSource = "http://developer.download.nvidia.com/compute/redist/cudnn/v5.1"

    @( @{ShortName = "CUDNN5180"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath },
                           @{Function = "VerifyDirectory"; Path = $envValue },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
         Download = @( @{Function = "DownloadForPlatform"; Platform = "^Microsoft Windows 7"; Source = "$downloadSource/$cudnnWin7"; Destination = "$cache\$cudnnWin7" },
                       @{Function = "DownloadForPlatform"; Platform = "^Microsoft Windows (8|10|Server 2008 R2|Server 2012 R2)"; Source = "$downloadSource/$cudnnWin10"; Destination = "$cache\$cudnnWin10" } );
         Action = @( @{Function = "ExtractAllFromZipForPlatform"; Platform = "^Microsoft Windows 7"; zipFileName = "$cache\$cudnnWin10"; destination = $targetFolder; destinationFolder = $prodSubDir },
                     @{Function = "ExtractAllFromZipForPlatform"; Platform = "^Microsoft Windows (8|10|Server 2008 R2|Server 2012 R2)"; zipFileName = "$cache\$cudnnWin10"; destination = $targetFolder; destinationFolder = $prodSubDir },
                     @{Function = "SetEnvironmentVariable"; EnvVar = $envVar; Content  = $envValue } );
         })
}

function OpOpenCV31(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "OpenCV-3.1"
    $prodFile = "opencv-3.1.0.exe"
    $prodSubDir = "OpenCV310"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "OPENCV_PATH_V31";
    $envValue = "$targetPath\build"
    $downloadSource = "https://netcologne.dl.sourceforge.net/project/opencvlibrary/opencv-win/3.1.0/opencv-3.1.0.exe"

    @(  @{ShortName = "OPENCV310"; Name = $prodName; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Please perform a manual installation of $prodName from $cache"; 
          Verification = @( @{Function = "VerifyFile"; Path = "$cache\$prodFile" } );
          Download = @( @{ Function = "Download"; Source = $downloadSource; Destination = "$cache\$prodFile" } );
         } )
}

function OpOpenCVInternal(
    [parameter(Mandatory=$true)][string] $server,
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "OpenCV-3.1"
    $prodFile = "opencv-3.1.0.zip"
    $prodSubDir = "Opencv3.1.0"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "OPENCV_PATH_V31";
    $envValue = "$targetPath\build"
    $downloadSource = "$server\$prodFile"

    @(  @{ShortName = "OPENCV310"; Name = $prodName; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName"; 
          Verification = @( @{Function = "VerifyDirectory"; Path = "$targetPath" },
                            @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
          Download = @( @{ Function = "LocalCopyFile"; Source = $downloadSource; Destination = "$cache\$prodFile" } );
          Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = "$targetFolder"; destinationFolder = $prodSubDir; zipSubTree= $prodSubDir },
                      @{Function = "SetEnvironmentVariable"; EnvVar= $envVar; Content = $envValue } );
         } )
}

function OpProtoBuf310VS15(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    # unzip protobuf source in $protoSourceDir = $targetfolder\src\$prodsubdir
    # create batch file to build protobuf files in $scriptDirectory = $targetFolder\script
    # the script file can be used to create the compiled protobuf libraries in $targetPath = $targetFolder\$prodSubDir

    $prodName = "ProtoBuf 3.1.0 Source"
    $prodFile = "protobuf310.zip"
    $prodName = "protobuf-3.1.0"
    $prodSubDir =  "protobuf-3.1.0-vs15"
    $batchFile = "buildProto.cmd"

    $protoSourceDir = join-path $targetFolder "src"
    $scriptDirectory = join-path $targetFolder "script"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "PROTOBUF_PATH"
    $envValue = $targetPath
    $downloadSource = "https://github.com/google/protobuf/archive/v3.1.0.zip"

    @( @{ShortName = "PROTO310VS15"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName"; 
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath } );
         Download = @( @{ Function = "Download"; Source = $downloadSource; Destination = "$cache\$prodFile" } );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $protoSourceDir; zipSubTree =$prodName; destinationFolder =$prodName },
                     @{Function = "MakeDirectory"; Path = $scriptDirectory },
                     @{Function = "CreateBuildProtobufBatch"; FileName = "$scriptDirectory\$batchFile"; SourceDir = (join-path $protoSourceDir $prodName); TargetDir = $targetPath } );
        } )
}

function OpProtoBuf310VS15Internal(
    [parameter(Mandatory=$true)][string] $server,
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "ProtoBuf 3.1.0 Prebuild VS2015"
    $prodFile = "PreBuildProtobuf310vs15.zip"
    $prodSubDir =  "protobuf-3.1.0-vs15"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "PROTOBUF_PATH"
    $envValue = $targetPath
    $downloadSource = "$server\$prodFile"

    @( @{ShortName = "PROTO310VS15PRE"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName"; 
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
         Download = @( @{ Function = "Download"; Source = $downloadSource; Destination = "$cache\$prodFile" } );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $targetFolder; zipSubTree ="protobuf-3.1.0-vs15"; destinationFolder = $prodSubDir },
                     @{Function = "SetEnvironmentVariable"; EnvVar = $envVar; Content  = $envValue } );
        } )
}

function OpScanProgram()
{
    @{ ShortName = "SCANPROG"; VerifyInfo = "Scan System for installed programs";
      Verification = @( @{Function = "VerifyScanPrograms" } )
     }
}

function OpSwig3010(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "SWIG 3.0.10"
    $prodFile = "swigwin-3.0.10.zip"
    $prodSubDir =  "swigwin-3.0.10"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "SWIG_PATH"
    $envValue = $targetPath
    $downloadSource = "http://prdownloads.sourceforge.net/project/swig/swigwin/swigwin-3.0.10/swigwin-3.0.10.zip"

    @( @{ShortName = "SWIG3010"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";  
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
         Download = @( @{ Function = "Download"; Source = $downloadSource; Destination = "$cache\$prodFile" } );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $targetFolder; zipSubTree =$prodSubDir; destinationFolder =$prodSubDir },
                     @{Function = "SetEnvironmentVariable"; EnvVar = $envVar; Content  = $envValue } );
        } )
}

function OpSysinternals(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "Sysinternal Suite"
    $prodFile = "SysinternalSuite.zip"
    $prodDependsFile = "depends22_x64.zip"
    $prodSubDir =  "SysInternal"
    $targetPath = join-path $targetFolder $prodSubDir
    $downloadSource = "https://download.sysinternals.com/files/SysinternalsSuite.zip"
    $downloadDependsSource = "http://dependencywalker.com/depends22_x64.zip"

    
    @( @{ShortName = "SYSINTERNAL"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName, dependency walker";  
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath }, 
                           @{Function = "VerifyFile"; Path = "$targetPath\depends.exe" },
                           @{Function = "VerifyPathIncludes"; Path = $targetPath } );
         Download = @( @{ Function = "Download"; Source = $downloadSource; Destination = "$cache\$prodFile" },
                       @{ Function = "Download"; Source = $downloadDependsSource; Destination = "$cache\$prodDependsFile" } );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $targetFolder; destinationFolder =$prodSubDir },
                     @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodDependsFile"; destination = $targetFolder; destinationFolder =$prodSubDir; AddToDirectory=$true },
                     @{Function = "AddToPath"; Dir = $targetPath; AtStart  = $false; }
                     @{Function = "SetRegistryKey"; Elevated = $true; Key = "registry::HKEY_CURRENT_USER\SOFTWARE\Sysinternals" },
                     @{Function = "SetRegistryKeyNameData"; Elevated = $true; Key = "registry::HKEY_CURRENT_USER\SOFTWARE\Sysinternals\Autologon"; RegName  = "EulaAccepted"; data = 1; dataType = "DWord" },
                     @{Function = "SetRegistryKeyNameData"; Elevated = $true; Key = "registry::HKEY_CURRENT_USER\SOFTWARE\Sysinternals\Handle"; RegName  = "EulaAccepted"; data = 1; dataType = "DWord" },
                     @{Function = "SetRegistryKeyNameData"; Elevated = $true;  Key = "registry::HKEY_CURRENT_USER\SOFTWARE\Sysinternals\ProcDump"; RegName = "EulaAccepted"; data = 1; dataType = "DWord" } )
        } )
}

function OpTestData(
    [parameter(Mandatory=$true)][string] $targetFolder,
    [parameter(Mandatory=$true)][string] $remoteData)
{
    $prodName = "Testdata Environment"
    @( @{ShortName = "TESTDATA"; VerifyInfo = "Checking for $prodName"; ActionInfo = "Setting up environment variable for $prodName";  
         Verification = @( @{Function = "VerifyEnvironment"; EnvVar = "CNTK_EXTERNAL_TESTDATA_REMOTE_DIRECTORY" },
                           @{Function = "VerifyEnvironment"; EnvVar = "CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY" } );
         Action = @( @{Function = "SetEnvironmentVariable"; EnvVar = "CNTK_EXTERNAL_TESTDATA_REMOTE_DIRECTORY"; Content = $remoteData },
                      @{Function = "SetEnvironmentVariable"; EnvVar = "CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY"; Content = $targetFolder } );
        } )
}

function OpAddVS12Runtime([parameter(Mandatory=$true)][string] $cache)
{
    $prodName = "VS2012 Runtime"

    @( @{ShortName = "VS2012"; VerifyInfo = "Checking for $prodName"; ActionInfo = "Installing $prodName"; 
         Verification = @( @{Function = "VerifyWinProductExists"; Match = "^Microsoft Visual C\+\+ 2012 x64 Additional Runtime" },
                            @{Function = "VerifyWinProductExists"; Match = "^Microsoft Visual C\+\+ 2012 x64 Minimum Runtime" } );
         Download = @( @{ Function = "Download"; Source = "http://download.microsoft.com/download/1/6/B/16B06F60-3B20-4FF2-B699-5E9B7962F9AE/VSU_4/vcredist_x64.exe"; Destination = "$cache\VSRuntime\11\vcredist_x64.exe" } );
         Action = @( @{Function = "InstallExe"; Command  = "$cache\VSRuntime\11\vcredist_x64.exe"; Param = "/install /passive /norestart" } )
         } )
}

function OpAddVS13Runtime([parameter(Mandatory=$true)][string] $cache)
{
    $prodName = "VS2013 Runtime" 

    @( @{ShortName = "VS2013"; VerifyInfo = "Checking for $prodName"; ActionInfo = "Installing $prodName"; 
          Verification = @( @{Function = "VerifyWinProductExists"; Match = "^Microsoft Visual C\+\+ 2013 x64 Additional Runtime" },
                            @{Function = "VerifyWinProductExists"; Match = "^Microsoft Visual C\+\+ 2013 x64 Minimum Runtime" } ); 
          Download = @( @{ Function = "Download"; Source = "http://download.microsoft.com/download/2/E/6/2E61CFA4-993B-4DD4-91DA-3737CD5CD6E3/vcredist_x64.exe"; Destination = "$localCache\VSRuntime\12\vcredist_x64.exe" } );
          Action = @( @{Function = "InstallExe"; Command  = "$localCache\VSRuntime\12\vcredist_x64.exe"; Param = "/install /passive /norestart" } ) 
         } )
}

function OpCheckVS15Update3
{
    @( @{Name = "Verify Installation of VS2015, Update 3"; ShortName = "PREVS15U3"; VerifyInfo = "Checking for Visual Studio 2015, Update 3"; 
                        Verification = @( @{Function = "VerifyWinProductVersion"; Match = "^Microsoft Build Tools 14.0 \(amd64\)$"; Version = "14.0.25420"; MatchExact = $true} ); 
                        PreReq = @( @{Function = "PrereqInfoVS15" } );
                        Action = @( @{Function = "StopInstallation" } )
                        } )
}

function OpCheckCuda8
{
    $programPath = join-path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA\v8.0"
    @( @{Name = "Verify Installation of NVidia Cuda 8"; ShortName = "PRECUDA8"; VerifyInfo = "Checking for NVidia Cuda 8"; 
         Verification = @( @{Function = "VerifyDirectory"; Path = $programPath },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = "CUDA_PATH_V8_0"; Content = $programPath } ); 
         PreReq = @( @{Function = "PrereqInfoCuda8" } );
         Action = @( @{Function = "StopInstallation" } )
        } )
}

function OpZlib(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "ZLib128"
    $prodFile = "zlib128.zip"
    $prodSubDir =  "zlib-1.2.8"
    $targetPath = join-path $targetFolder $prodSubDir
    $downloadSource = "http://zlib.net/zlib128.zip"

    @( @{ShortName = "ZLIN128"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";  
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath } );
         Download = @( @{ Function = "Download"; Source = $downloadSource; Destination = "$cache\$prodFile" } );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $targetFolder; zipSubTree =$prodSubDir; destinationFolder =$prodSubDir } );
        } )
}

function OpLibzip(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "Libzip-1.1.3"
    $prodFile = "libzip-1.1.3.tar.gz"
    $prodSubDir =  "libzip-1.1.3"
    $targetPath = join-path $targetFolder $prodSubDir
    $downloadSource = "https://nih.at/libzip/libzip-1.1.3.tar.gz"

    @( @{ShortName = "LIBZIP113"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";  
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath } );
         Download = @( @{ Function = "Download"; Source = $downloadSource; Destination = "$cache\$prodFile" } );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $targetFolder; zipSubTree =$prodSubDir; destinationFolder =$prodSubDir } );
        } )
}


function OpZLibInternal(
    [parameter(Mandatory=$true)][string] $server,
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "ZLib Precompiled"
    $prodFile = "zlib.zip"
    $prodSubDir =  "zlib"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "ZLIB_PATH"
    $envValue = $targetPath
    $downloadSource = "$server\$prodFile"

    @( @{ShortName = "LIBZIP113"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";  
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue }  );
         Download = @( @{ Function = "Download"; Source = $downloadSource; Destination = "$cache\$prodFile" } );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $targetFolder; destinationFolder =$prodSubDir },
                     @{Function = "SetEnvironmentVariable"; EnvVar = $envVar; Content  = $envValue }  );
        } )
}

function OpZlibVS15(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    # unzip protobuf source in $protoSourceDir = $targetfolder\src\$prodsubdir
    # create batch file to build protobuf files in $scriptDirectory = $targetFolder\script
    # the script file can be used to create the compiled protobuf libraries in $targetPath = $targetFolder\$prodSubDir

    $prodName = "zlib / libzip from source"
    $zlibProdName = "zlib-1.2.8"
    $zlibFilename = "zlib128.zip" 
    $zlibDownloadSource = "http://zlib.net/zlib128.zip"
    $libzipProdName = "libzip-1.1.3"
    $libzipFilename = "libzip-1.1.3.tar.gz" 
    $libzipDownloadSource = "https://nih.at/libzip/libzip-1.1.3.tar.gz"
    
    $prodSubDir =  "zlib-vs15"
    $batchFile = "buildZlibVS15.cmd"

    $sourceCodeDir = join-path $targetFolder "src"
    $scriptDirectory = join-path $targetFolder "script"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "ZLIB_PATH"
    $envValue = $targetPath
    
    @( @{ShortName = "ZLIBVS15"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName"; 
         Verification = @( @{Function = "VerifyDirectory"; Path = "$sourceCodeDir\$zlibProdName" },
                           @{Function = "VerifyDirectory"; Path = "$sourceCodeDir\$libzipProdName" },
                           @{Function = "VerifyFile"; Path = "$scriptDirectory\$batchFile" } );
         Download = @( @{ Function = "Download"; Source = $zlibDownloadSource; Destination = "$cache\$zlibFilename" }, 
                       @{ Function = "Download"; Source = $libzipDownloadSource; Destination = "$cache\$libzipFilename" } );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$zlibFilename"; destination = $sourceCodeDir; zipSubTree =$zlibProdName; destinationFolder =$zlibProdName },
                     @{Function = "ExtractAllFromTarGz"; SourceFile =  "$cache\$libzipFilename"; TargzFileName = "$libzipFilename"; destination = $sourceCodeDir },
                     @{Function = "MakeDirectory"; Path = $scriptDirectory },
                     @{Function = "CreateBuildZlibBatch"; FileName = "$scriptDirectory\$batchFile"; zlibSourceDir = (join-path $sourceCodeDir $zlibProdName); libzipSourceDir = (join-path $sourceCodeDir $libzipProdName); TargetDir = $targetPath } );
        } )
}


# vim:set expandtab shiftwidth=2 tabstop=2: