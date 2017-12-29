# file: runme.tcl

catch { load ./example[info sharedlibextension] example}

set a 37
set b 42

# Now call our C function with a bunch of callbacks

puts "Trying some C callback functions"
puts "    a        = $a"
puts "    b        = $b"
puts "    ADD(a,b) = [do_op $a $b $ADD]"
puts "    SUB(a,b) = [do_op $a $b $SUB]"
puts "    MUL(a,b) = [do_op $a $b $MUL]"

puts "Here is what the C callback function objects look like in Tcl"
puts "    ADD      = $ADD"
puts "    SUB      = $SUB"
puts "    MUL      = $MUL"

