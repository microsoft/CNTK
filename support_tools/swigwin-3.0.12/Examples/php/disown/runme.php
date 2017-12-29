<?php

# This file illustrates the low-level C++ interface
# created by SWIG.  In this case, all of our C++ classes
# get converted into function calls.

require("example.php");

# ----- Object creation -----

print "Creating some objects:\n";
$c = new Circle(10);
print "    Created circle \$c\n";
$s = new Square(10);
print "    Created square \$s\n";

# ----- Create the ShapeContainer ----

$container = new ShapeContainer();

$container->addShape($c);
$container->addShape($s);

# ----- Access a static member -----

print "\nA total of " . Shape::nshapes() . " shapes were created\n";

# ----- Delete by the old references -----
# This should not truely delete the shapes because they are now owned
# by the ShapeContainer.

print "Delete the old references.";

# Note: this invokes the virtual destructor
$c = NULL;
$s = NULL;

print "\nA total of " . Shape::nshapes() . " shapes remain\n";

# ----- Delete by the container -----
# This should truely delete the shapes

print "Delete the container.";
$container = NULL;
print "\nA total of " . Shape::nshapes() . " shapes remain\n";

print "Goodbye\n";

?>
