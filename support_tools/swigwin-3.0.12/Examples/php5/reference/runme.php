<?php

# This file illustrates the manipulation of C++ references in PHP.

require "example.php";

# ----- Object creation -----

print "Creating some objects:\n";
$a = new Vector(3, 4, 5);
$b = new Vector(10, 11, 12);

print "    Created a: {$a->as_string()}\n";
print "    Created b: {$b->as_string()}\n";

# ----- Call an overloaded operator -----

# This calls the wrapper we placed around
#
#      operator+(const Vector &a, const Vector &) 
#
# It returns a new allocated object.

print "Adding a+b\n";
$c = example::addv($a, $b);
print "    a+b ={$c->as_string()}\n";

# ----- Create a vector array -----

print "Creating an array of vectors\n";
$va = new VectorArray(10);

print "    va: size={$va->size()}\n";

# ----- Set some values in the array -----

# These operators copy the value of $a and $b to the vector array
$va->set(0, $a);
$va->set(1, $b);
$va->set(2, addv($a, $b));

# Get some values from the array

print "Getting some array values\n";
for ($i = 0; $i < 5; $i++) {
    print "    va[$i] = {$va->get($i)->as_string()}\n";
}

?>
