if [ catch { load ./reference_global_vars[info sharedlibextension] reference_global_vars} err_msg ] {
	puts stderr "Could not load shared object:\n$err_msg"
}

# const class reference variable
if {[ [getconstTC ] cget -num] != 33 } {
   puts stderr "test 1 failed"
   exit 1
}

# primitive reference variables
set var_bool [createref_bool 0]
if {[value_bool $var_bool] != 0} {
   puts stderr "test 2 failed"
   exit 1
}

set var_bool [createref_bool 1]
if {[value_bool $var_bool] != 1} {
   puts stderr "test 3 failed"
   exit 1
}

set var_char [createref_char "w"]
if {[value_char $var_char] != "w"} {
   puts stderr "test 4 failed"
   exit 1
}

set var_unsigned_char [createref_unsigned_char 10]
if {[value_unsigned_char $var_unsigned_char] != 10} {
   puts stderr "test 5 failed"
   exit 1
}

set var_signed_char [createref_signed_char 10]
if {[value_signed_char $var_signed_char] != 10} {
   puts stderr "test 6 failed"
   exit 1
}

set var_short [createref_short 10]
if {[value_short $var_short] != 10} {
   puts stderr "test 7 failed"
   exit 1
}

set var_unsigned_short [createref_unsigned_short 10]
if {[value_unsigned_short $var_unsigned_short] != 10} {
   puts stderr "test 8 failed"
   exit 1
}

set var_int [createref_int 10]
if {[value_int $var_int] != 10} {
   puts stderr "test 9 failed"
   exit 1
}

set var_unsigned_int [createref_unsigned_int 10]
if {[value_unsigned_int $var_unsigned_int] != 10} {
   puts stderr "test 10 failed"
   exit 1
}

set var_long [createref_long 10]
if {[value_long $var_long] != 10} {
   puts stderr "test 11 failed"
   exit 1
}

set var_unsigned_long [createref_unsigned_long 10]
if {[value_unsigned_long $var_unsigned_long] != 10} {
   puts stderr "test 12 failed"
   exit 1
}

set var_long_long [createref_long_long 10]
if {[value_long_long $var_long_long] != 10} {
   puts stderr "test 13 failed"
   exit 1
}

set var_unsigned_long_long [createref_unsigned_long_long 10]
if {[value_unsigned_long_long $var_unsigned_long_long] != 10} {
   puts stderr "test 14 failed"
   exit 1
}

set var_float [createref_float 10.5]
if {[value_float $var_float] != 10.5} {
   puts stderr "test 15 failed"
   exit 1
}

set var_double [createref_double 10.5]
if {[value_double $var_double] != 10.5} {
   puts stderr "test 16 failed"
   exit 1
}

# class reference variable
set var_TestClass [createref_TestClass [TestClass tc 20] ]
if {[ [value_TestClass $var_TestClass] cget -num] != 20} {
   puts stderr "test 17 failed"
   exit 1
}

