require 'example'

a = 37
b = 42

# Now call our C function with a bunch of callbacks

puts "Trying some C callback functions"
puts "    a        = #{a}"
puts "    b        = #{b}"
puts "    ADD(a,b) = #{Example.do_op(a,b,Example::ADD)}"
puts "    SUB(a,b) = #{Example.do_op(a,b,Example::SUB)}"
puts "    MUL(a,b) = #{Example.do_op(a,b,Example::MUL)}"

puts "Here is what the C callback function objects look like in Ruby"
puts "    ADD      = #{Example::ADD}"
puts "    SUB      = #{Example::SUB}"
puts "    MUL      = #{Example::MUL}"

puts "Call the functions directly..."
puts "    add(a,b) = #{Example.add(a,b)}"
puts "    sub(a,b) = #{Example.sub(a,b)}"
