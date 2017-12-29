# Primitive ref testcase.  Tests to make sure references to 
# primitive types are passed by value

if [ catch { load ./primitive_ref[info sharedlibextension] primitive_ref} err_msg ] {
	puts stderr "Could not load shared object:\n$err_msg"
}

if { [ref_int 3] != 3 } { puts stderr "ref_int failed" }
if { [ref_uint 3] != 3 } { puts stderr "ref_uint failed" }
if { [ref_short 3] != 3 } { puts stderr "ref_short failed" }
if { [ref_ushort 3] != 3 } { puts stderr "ref_ushort failed" }
if { [ref_long 3] != 3 } { puts stderr "ref_long failed" }
if { [ref_ulong 3] != 3 } { puts stderr "ref_ulong failed" }
if { [ref_schar 3] != 3 } { puts stderr "ref_schar failed" }
if { [ref_uchar 3] != 3 } { puts stderr "ref_uchar failed" }
if { [ref_float 3.5] != 3.5 } { puts stderr "ref_float failed" }
if { [ref_double 3.5] != 3.5 } { puts stderr "ref_double failed" }
if { [ref_char x] != "x" } { puts stderr "ref_char failed" }

