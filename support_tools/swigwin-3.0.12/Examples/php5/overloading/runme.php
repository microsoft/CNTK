<?php

# This file illustrates the low-level C++ interface
# created by SWIG.  In this case, all of our C++ classes
# get converted into function calls.

include("example.php");

# ----- Object creation -----

print "Creating some objects:\n";
$c = new Circle(10);
print "    Created circle \$c\n";
$s = new Square(10);
print "    Created square \$s\n";

# ----- Access a static member -----

print "\nA total of " . Shape::nshapes() . " shapes were created\n";

# ----- Member data access -----

# Set the location of the object.
# Note: methods in the base class Shape are used since
# x and y are defined there.

$c->x = 20;
$c->y = 30;
$s->x = -10;
$s->y = 5;

print "\nHere is their current position:\n";
print "    Circle = (" . $c->x . "," . $c->y . ")\n";
print "    Square = (" . $s->x . "," . $s->y . ")\n";

# ----- Call some methods -----

print "\nCall some overloaded methods:\n";
foreach (array(1, 2.1, "quick brown fox", $c, $s) as $o) {
  print "        overloaded = " .  overloaded($o) . "\n";
}

# Need to unset($o) or else we hang on to a reference to the Square object.
unset($o);

# ----- Delete everything -----

print "\nGuess I'll clean up now\n";

# Note: this invokes the virtual destructor
unset($c);
$s = 42;

print Shape::nshapes() . " shapes remain\n";

print "Goodbye\n";

?>
