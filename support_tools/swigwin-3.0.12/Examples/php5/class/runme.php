<?php

# This example illustrates how member variables are wrapped.

require("example.php");

# ----- Object creation -----

print "Creating some objects:\n";
$c = new Circle(10);
print "    Created circle\n";
$s = new Square(10);
print "    Created square\n";

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
print "    Circle = ({$c->x},{$c->y})\n";
print "    Square = ({$s->x},{$s->y})\n";

# ----- Call some methods -----

# Notice how the Shape_area() and Shape_perimeter() functions really
# invoke the appropriate virtual method on each object.
print "\nHere are some properties of the shapes:\n";
foreach (array($c,$s) as $o) {
      print "    ". get_class($o) . "\n";
      print "        area      = {$o->area()}\n";
      print "        perimeter = {$o->perimeter()}\n";
}

# ----- Delete everything -----

print "\nGuess I'll clean up now\n";

# Note: this invokes the virtual destructor
$c = NULL;
$s = NULL;

# and don't forget the $o from the for loop above.  It still refers to
# the square.
$o = NULL;

print Shape::nshapes() . " shapes remain\n";
print "Goodbye\n";

?>
