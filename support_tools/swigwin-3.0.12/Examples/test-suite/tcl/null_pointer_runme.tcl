if [ catch { load ./null_pointer[info sharedlibextension] null_pointer} err_msg ] {
	puts stderr "Could not load shared object:\n$err_msg"
}

set a [A]
if {[func $a] != 0} {
    puts stderr "null_pointer test 1 failed"
    exit 1
}

set null [getnull]
if {$null != "NULL"} {
    puts stderr "null_pointer test 2 failed"
    exit 1
}

if {[llength [info commands "NULL"]] != 0} {
    puts stderr "null_pointer test 3 failed"
    exit 1
}

