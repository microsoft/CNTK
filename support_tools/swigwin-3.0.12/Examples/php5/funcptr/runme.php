<?php

require "example.php";

$a = 37;
$b = 42;

# Now call our C function with a bunch of callbacks

print "Trying some C callback functions\n";
print "    a        = $a\n";
print "    b        = $b\n";
print "    ADD(a,b) = ". do_op($a,$b,ADD)."\n";
print "    SUB(a,b) = ". do_op($a,$b,SUB)."\n";
print "    MUL(a,b) = ". do_op($a,$b,MUL)."\n";

print "Here is what the C callback function objects look like in php\n";
print "Using swig style string pointers as we need them registered as constants\n";
print "    ADD      = " . ADD . "\n";
print "    SUB      = " . SUB . "\n";
print "    MUL      = " . MUL . "\n";

?>

