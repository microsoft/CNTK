<?php

require "example.php";

# Call our gcd() function

$x = "42 aaa";
$y = 105;
$g = gcd($x,$y);
print "The gcd of $x and $y is $g\n";

# Manipulate the Foo global variable

# Output its current value
print "Foo = " . Foo_get() . "\n";

# Change its value
Foo_set(3.1415926);

# See if the change took effect ( this isn't a good example for php, see
#				  manual for why. )
print "Foo = "  . Foo_get() . "\n";
print_Foo();

?>
