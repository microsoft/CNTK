# file: runme.rb

require 'example'

a = 37
b = 42

# Now call our C function with a bunch of callbacks

print "Trying some C callback functions\n"
print "    a        = #{a}\n"
print "    b        = #{b}\n"
print "    ADD(a,b) = ", Example::do_op(a,b,Example::ADD),"\n"
print "    SUB(a,b) = ", Example::do_op(a,b,Example::SUB),"\n"
print "    MUL(a,b) = ", Example::do_op(a,b,Example::MUL),"\n"

print "Here is what the C callback function objects look like in Ruby\n"
print "    ADD      = #{Example::ADD}\n"
print "    SUB      = #{Example::SUB}\n"
print "    MUL      = #{Example::MUL}\n"

