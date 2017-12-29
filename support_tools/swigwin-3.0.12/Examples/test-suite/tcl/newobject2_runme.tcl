if [ catch { load ./newobject2[info sharedlibextension] newobject2} err_msg ] {
	puts stderr "Could not load shared object:\n$err_msg"
}

set foo1 [makeFoo]
if {[fooCount] != 1} {
    puts stderr "newobject2 test 1 failed"
    exit 1
}

set foo2 [makeFoo]
if {[fooCount] != 2} {
    puts stderr "newobject2 test 2 failed"
    exit 1
}

#$foo1 -delete
#if {[fooCount] != 1} {
#    puts stderr "newobject2 test 3 failed"
#    exit 1
#}

#$foo2 -delete
#if {[fooCount] != 0} {
#    puts stderr "newobject2 test 4 failed"
#    exit 1
#}
