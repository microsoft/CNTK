
if [ catch { load ./bools[info sharedlibextension] bools} err_msg ] {
	puts stderr "Could not load shared object:\n$err_msg"
}

# bool constant check
if {$constbool != 0} {
    puts stderr "Runtime test 1 failed"
    exit 1
}

# bool variables check
if {$bool1 != 1} {
    puts stderr "Runtime test 2 failed"
    exit 1
}

if {$bool2 != 0} {
    puts stderr "Runtime test 3 failed"
    exit 1
}

if { [ value $pbool ] != $bool1} {
    puts stderr "Runtime test 4 failed"
    exit 1
}

if { [ value $rbool ] != $bool2} {
    puts stderr "Runtime test 5 failed"
    exit 1
}

if { [ value $const_pbool ] != $bool1} {
    puts stderr "Runtime test 6 failed"
    exit 1
}

if { $const_rbool != $bool2} {
    puts stderr "Runtime test 7 failed"
    exit 1
}

# bool functions check
if { [ bo 0 ] != 0} {
    puts stderr "Runtime test 8 failed"
    exit 1
}

if { [ bo 1 ] != 1} {
    puts stderr "Runtime test 9 failed"
    exit 1
}

if { [ value  [ rbo $rbool ] ] !=  [ value $rbool ]} {
    puts stderr "Runtime test 10 failed"
    exit 1
}

if { [ value  [ pbo $pbool ] ] !=  [ value $pbool ]} {
    puts stderr "Runtime test 11 failed"
    exit 1
}

if { [ const_rbo $const_rbool ] !=   $const_rbool } {
    puts stderr "Runtime test 12 failed"
    exit 1
}

if { [ value  [ const_pbo $const_pbool ] ] !=  [ value $const_pbool ]} {
    puts stderr "Runtime test 13 failed"
    exit 1
}

