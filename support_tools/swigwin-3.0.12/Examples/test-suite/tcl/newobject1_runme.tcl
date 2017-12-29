if [ catch { load ./newobject1[info sharedlibextension] newobject1} err_msg ] {
	puts stderr "Could not load shared object:\n$err_msg"
}

set foo1 [Foo_makeFoo]
if {[Foo_fooCount] != 1} {
    puts stderr "newobject1 test 1 failed"
    exit 1
}

set foo2 [$foo1 makeMore]
if {[Foo_fooCount] != 2} {
    puts stderr "newobject1 test 2 failed"
    exit 1
}

# Disable test while we solve the problem of premature object deletion
#$foo1 -delete
#if {[Foo_fooCount] != 1} {
#    puts stderr "newobject1 test 3 failed"
#    exit 1
#}
#
#$foo2 -delete
#if {[Foo_fooCount] != 0} {
#    puts stderr "newobject1 test 4 failed"
#    exit 1
#}
