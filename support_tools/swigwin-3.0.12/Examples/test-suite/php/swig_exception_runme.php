<?php

require("swig_exception.php");
require("tests.php");

$c = new Circle(10);
$s = new Square(10);

if (Shape::nshapes() != 2) {
    check::fail("Shape::nshapes() should be 2, actually ".Shape::nshapes());
}

# ----- Throw exception -----
try {
    $c->throwException();
    check::fail("Exception wasn't thrown");
} catch (Exception $e) {
    if ($e->getMessage() != "OK") {
	check::fail("Exception getMessage() should be \"OK\", actually \"".$e->getMessage()."\"");
    }
}

# ----- Delete everything -----

$c = NULL;
$s = NULL;
$e = NULL;

if (Shape::nshapes() != 0) {
    check::fail("Shape::nshapes() should be 0, actually ".Shape::nshapes());
}

?>
