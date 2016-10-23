$operations = @(
    @{Name = "Installation VS2012 Runtime"; ShortName = "VS2012"; Info = "Install VS2012 Runtime"; 
      Verification = @( @{Function = "VerifyRegistryKeyNameData"; Key = "Registry::HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\11.0\VC\Runtimes\x64"; RegName = "Installed"; RegData  = "1" } );
      Action = @( @{Function = "InstallExe"; Command  = "$cntkRootDir\prerequisites\VS2012\vcredist_x64.exe"; Param = "/install /passive /norestart"; Message="Installing VS2012 Runtime...." } )
     },
    @{Name = "Installation VS2013 Runtime"; ShortName = "VS2013"; Info = "Install VS2013 Runtime";
      Verification = @( @{Function = "VerifyRegistryKeyNameData"; Key = "Registry::HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\12.0\VC\Runtimes\x64"; RegName = "Installed"; RegData  = "1" } );
      Action = @( @{Function = "InstallExe"; Command  = "$cntkRootDir\prerequisites\VS2013\vcredist_x64.EXE"; Param = "/install /passive /norestart"; Message="Installing VS2013 Runtime...." } )
     },
    @{Name = "MSMPI Installation"; ShortName = "CNTK"; Info = "Install MSMPI";
      Verification = @( @{Function = "VerifyRegistryKeyName"; Key = "Registry::HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\MPI"; RegName = "Version" } );
      Action = @( @{Function = "InstallExe"; Command = "$cntkRootDir\prerequisites\msmpisetup.EXE"; Param = "/unattend"; Message="Installing MSMPI ...." } )
     },
    @{Name = "Anaconda3-4.1.1"; ShortName = "ANA3-411"; Info = "Install Anaconda3-4.1.10";
      Verification = @( @{Function = "VerifyDirectory"; Path = "$AnacondaBasePath"; } );
      Download = @( @{Function = "Download"; Source = "https://repo.continuum.io/archive/Anaconda3-4.1.1-Windows-x86_64.exe"; Destination = "$localCache\Anaconda3-4.1.1-Windows-x86_64.exe" } );
      Action = @( @{Function = "InstallExe"; Command = "$localCache\Anaconda3-4.1.1-Windows-x86_64.exe"; Param = "/InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=$AnacondaBasePath"; runAs=$false; Message="Installing Anaconda3-4.1.1. This will take several minutes. Please be patient ...."} );
     },
    @{Name = "CNTK Python Environment 3.4"; ShortName = "CNTKPY34"; Info = "Setup CNTK PythonEnvironment 3.4";
      Verification  = @( @{Function = "VerifyDirectory"; Path = "$AnacondaBasePath\envs\cntk-py34"; } );
      Action = @( @{Function = "InstallExe"; Command = "$AnacondaBasePath\Scripts\conda.exe"; Param = "env create --file $MyDir\conda-windows-cntk-py34-environment.yml --prefix c:\local\Anaconda3-4.1.1-Windows-x86_64\envs\cntk-py34"; WorkDir = "$AnacondaBasePath\Scripts"; runAs=$false; Message="Setting up CNTK-PY34 environment. Please be patient...." } )
     },
    @{Name = "CNTK WHL Install"; ShortName = "CNTKWHL34"; Info = "Setup CNTK Wheel";
      Verification  = @( @{Function = "VerifyWheelDirectory"; WheelDirectory = "$AnacondaBasePath\envs\cntk-py34\Lib\site-packages\cntk"; ForceUpdate = $ForceWheelUpdate} );
      Action = @( @{Function = "InstallWheel"; BasePath = "$AnacondaBasePath"; EnvName = "cntk-py34"; WheelDirectory="$AnacondaBasePath\envs\cntk-py34\Lib\site-packages\cntk"; ForceUpdate = $ForceWheelUpdate; Message="Setting up CNTK Wheel environment. Please be patient...." } )
     },
    @{Name = "Create CNTKPY34 batch file"; ShortName = "BATCH34"; Info = "Create CNTKPY34 batch file";
      Verification  = @( @{Function = "VerifyFile"; Path = "$cntkRootDir\scripts\cntkpy34.bat" } );
      Action = @( @{Function = "CreateBatch"; Filename = "$cntkRootDir\scripts\cntkpy34.bat"; Command = "$AnacondaBasePath\Scripts\activate cntk-py34" } )
     },
    @{Name = "Git"; ShortName = "GIT"; Info = "Install Git"; Target = "gitrepo";
      Verification  = @( @{Function = "VerifyRegistryKeyName"; Key = "Registry::HKEY_LOCAL_MACHINE\SOFTWARE\GitForWindows"; RegName = "CurrentVersion"; } );
      Download = @( @{Function = "Download"; Source = "https://github.com/git-for-windows/git/releases/download/v2.10.1.windows.1/Git-2.10.1-64-bit.exe"; Destination = "$localCache\Git-2.10.1-64-bit.exe" } );
      Action = @( @{Function = "InstallExe"; Command  = "$localCache\Git-2.10.1-64-bit.exe"; Param = "/SP- /SILENT /NORESTART"; Message="Installing Git. Please be patient...."},
                  @{Function = "AddToPath"; Dir = "C:\Program Files\Git\cmd"; AtStart  = $true; } )
     },
    @{Name = "Clone CNTK from Github"; ShortName = "CNTKCLONE"; Info = "Clone CNTK from Github repository";
      Verification = @( @{Function = "VerifyDirectory"; Path = "c:\repos\cntk" } );
      Action = @( @{Function = "MakeDirectory"; Path = "c:\repos" },
                  @{Function = "InstallExe"; Command = "C:\Program Files\Git\bin\git.exe"; Param = "clone --recursive https://github.com/Microsoft/CNTK/"; WorkDir = "c:\repos"; Message="Cloning CNTK repository...." }, 
                  @{Function = "InstallExe"; Command = "C:\Program Files\Git\bin\git.exe"; Param = "checkout $RepoTag"; WorkDir = "c:\repos\cntk"; Message="Check out alpha4 CNTK repository...." })
     }
)
