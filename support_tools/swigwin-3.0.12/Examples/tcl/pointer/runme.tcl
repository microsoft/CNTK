# file: runme.tcl

catch { load ./example[info sharedlibextension] example}

# First create some objects using the pointer library.
puts "Testing the pointer library"
set a [new_intp]
set b [new_intp]
set c [new_intp]       ;# Memory for result

intp_assign $a 37
intp_assign $b 42

puts "     a = $a"
puts "     b = $b"
puts "     c = $c"

# Call the add() function with some pointers
add $a $b $c

# Now get the result
set r [intp_value $c]
puts "     37 + 42 = $r"

# Clean up the pointers
delete_intp $a
delete_intp $b
delete_intp $c

# Now try the typemap library
# This should be much easier. Now how it is no longer
# necessary to manufacture pointers.

puts "Trying the typemap library"
set r [sub 37 42]
puts "     37 - 42 = $r"

# Now try the version with multiple return values

puts "Testing multiple return values"
set qr [divide 42 37]
set q [lindex $qr 0]
set r [lindex $qr 1]
puts "     42/37 = $q remainder $r"



