# file: runme.pl

use example;

# Call our gcd() function

$x = 42;
$y = 105;
$g = example::gcd($x,$y);
print "The gcd of $x and $y is $g\n";

# Call the gcdmain() function
@a = ("gcdmain","42","105");
example::gcdmain(\@a);

# Call the count function
print example::count("Hello World", "l"),"\n";

# Call the capitize function

print example::capitalize("hello world"),"\n";







