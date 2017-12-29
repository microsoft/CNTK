# file: runme.tcl
# Try to load as a dynamic module.

catch { load ./example[info sharedlibextension] example}

# Create a couple of a vectors

set v [new_Vector 1 2 3]
set w [new_Vector 10 11 12]

puts "I just created the following vectors"
vector_print $v
vector_print $w

# Now call some of our functions

puts "\nNow I'm going to compute the dot product"
set d [dot_product $v $w]
puts "dot product = $d (should be 68)"

# Add the vectors together

puts "\nNow I'm going to add the vectors together"
set r [vector_add $v $w]
vector_print $r
puts "The value should be (11,13,15)"

# Now I'd better clean up the return result r

puts "\nNow I'm going to clean up the return result"
free $r

puts "Good"






