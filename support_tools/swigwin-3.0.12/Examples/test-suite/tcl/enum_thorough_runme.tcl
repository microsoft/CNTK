
if [ catch { load ./enum_thorough[info sharedlibextension] enum_thorough} err_msg ] {
	puts stderr "Could not load shared object:\n$err_msg"
}

if { [speedTest0 $SpeedClass_slow] != $SpeedClass_slow } { puts stderr "speedTest0 failed" }
if { [speedTest4 $SpeedClass_slow] != $SpeedClass_slow } { puts stderr "speedTest4 failed" }
if { [speedTest5 $SpeedClass_slow] != $SpeedClass_slow } { puts stderr "speedTest5 failed" }

