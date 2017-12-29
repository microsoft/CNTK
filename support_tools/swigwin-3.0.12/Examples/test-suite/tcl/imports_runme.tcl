
# This is the imports runtime testcase. 
proc import {} {
    if [ catch { load ./imports_b[info sharedlibextension] imports_b} err_msg ] {
            puts stderr "Could not load shared object:\n$err_msg"
            exit 1
    }
    if [ catch { load ./imports_a[info sharedlibextension] imports_a} err_msg ] {
            puts stderr "Could not load shared object:\n$err_msg"
            exit 1
    }
}

import

set x [new_B]
A_hello $x
if [ catch { $x nonexistant } ] {
} else {
  puts stderr "nonexistant method did not throw exception\n"
  exit 1
}
