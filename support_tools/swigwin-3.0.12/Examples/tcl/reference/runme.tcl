# file: runme.tcl

# This file illustrates the manipulation of C++ references in Tcl

catch { load ./example[info sharedlibextension] example}

# ----- Object creation -----

puts "Creating some objects:"
set  a [new_Vector 3 4 5]
set  b [new_Vector 10 11 12]

puts "    Created [Vector_print $a]"
puts "    Created [Vector_print $b]"

# ----- Call an overloaded operator -----

# This calls the wrapper we placed around
#
#      operator+(const Vector &a, const Vector &) 
#
# It returns a new allocated object.

puts "Adding a+b"
set c [addv $a $b]
puts "    a+b = [Vector_print $c]"

# Note: Unless we free the result, a memory leak will occur
delete_Vector $c

# ----- Create a vector array -----

# Note: Using the high-level interface here
puts "Creating an array of vectors"
VectorArray va 10
puts "    va = [va cget -this]"


# ----- Set some values in the array -----

# These operators copy the value of $a and $b to the vector array
va set 0 $a
va set 1 $b

# This will work, but it will cause a memory leak!

va set 2 [addv $a $b]

# The non-leaky way to do it

set c [addv $a $b]
va set 3 $c
delete_Vector $c

# Get some values from the array

puts "Getting some array values"
for {set i 0} {$i < 5} {incr i 1} {
    puts "    va($i) = [Vector_print [va get $i]]"
}

# Watch under resource meter to check on this
puts "Making sure we don't leak memory."
for {set i 0} {$i < 1000000} {incr i 1} {
    set c [va get [expr {$i % 10}]]
}

# ----- Clean up -----
puts "Cleaning up"

rename va ""

delete_Vector $a
delete_Vector $b


