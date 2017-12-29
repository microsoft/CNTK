# file: runme.py

import example

a = 37
b = 42

# Now call our C function with a bunch of callbacks

print "Trying some C callback functions"
print "    a        =", a
print "    b        =", b
print "    ADD(a,b) =", example.do_op(a, b, example.ADD)
print "    SUB(a,b) =", example.do_op(a, b, example.SUB)
print "    MUL(a,b) =", example.do_op(a, b, example.MUL)

print "Here is what the C callback function objects look like in Python"
print "    ADD      =", example.ADD
print "    SUB      =", example.SUB
print "    MUL      =", example.MUL

print "Call the functions directly..."
print "    add(a,b) =", example.add(a, b)
print "    sub(a,b) =", example.sub(a, b)
