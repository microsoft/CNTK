#
# Perl5 script for testing simple example

use example;

# Call our gcd() function

$x = 42;
$y = 105;
$g = example::gcd($x,$y);
print "The gcd of $x and $y is $g\n";

# Manipulate the Foo global variable

# Output its current value
print "Foo = $example::Foo\n";

# Change its value
$example::Foo = 3.1415926;

# See if the change took effect
print "Foo = $example::Foo\n";

