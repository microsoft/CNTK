# file: runme.pl

use example;

$a = 37;
$b = 42;

# Now call our C function with a bunch of callbacks

print "Trying some C callback functions\n";
print "    a        = $a\n";
print "    b        = $b\n";
print "    ADD(a,b) = ", example::do_op($a,$b,$example::ADD),"\n";
print "    SUB(a,b) = ", example::do_op($a,$b,$example::SUB),"\n";
print "    MUL(a,b) = ", example::do_op($a,$b,$example::MUL),"\n";

print "Here is what the C callback function objects look like in Perl\n";
print "    ADD      = $example::ADD\n";
print "    SUB      = $example::SUB\n";
print "    MUL      = $example::MUL\n";

