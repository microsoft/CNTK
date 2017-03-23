$ThisModule = $MyInvocation.MyCommand.Path -replace '\.Tests\.ps1$'
$ThisModuleName = $ThisModule | Split-Path -Leaf

Get-Module -Name $ThisModuleName -All | Remove-Module -Force -ErrorAction Ignore

Import-Module -Name "$ThisModule.psm1" -Force -ErrorAction Stop

describe 'Invoke-DosCommand' {
  it 'runs something' {
    Invoke-DosCommand cmd.exe -Argument @('/c', 'exit', '0')
  }
  it 'throws' {
    { Invoke-DosCommand cmd.exe -Argument @('/c', 'exit', '1') } | Should Throw
  }
  it 'ignores non-zero exit code' {
    Invoke-DosCommand cmd.exe -IgnoreNonZeroExitCode -Argument @('/c', 'exit', '1')
  }
  it 'accepts maximum error level' {
    Invoke-DosCommand cmd.exe -MaxErrorLevel 1 -Argument @('/c', 'exit', '1')
  }
  it 'throws if maximum error level exceeded' {
    { Invoke-DosCommand cmd.exe -MaxErrorLevel 1 -Argument @('/c', 'exit', '2') } | Should Throw
  }
  it 'returns output' {
    Invoke-DosCommand cmd.exe -Argument @('/c', 'echo', 'Output') | Should BeExactly "Output"
  }
  it 'suppresses output' {
    Invoke-DosCommand cmd.exe -SuppressOutput -Argument @('/c', 'echo', 'Output') | Should BeNullOrEmpty
  }
  it 'changes directory' {
    Invoke-DosCommand cmd.exe -WorkingDirectory $env:TEMP -Argument @('/c', 'cd') | Should Be $env:TEMP
  }
}
