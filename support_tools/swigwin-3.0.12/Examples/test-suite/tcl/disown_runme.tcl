
# This is the union runtime testcase. It ensures that values within a 
# union embedded within a struct can be set and read correctly.

if [ catch { load ./disown[info sharedlibextension] disown} err_msg ] {
	puts stderr "Could not load shared object:\n$err_msg"
}

set x 0
while {$x<100} {
  set a [new_A]
  B b
  b acquire $a
  incr x
}

