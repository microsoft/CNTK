<?php
require "tests.php";
require "li_carrays_cpp.php";

// Check functions.
check::functions(array(new_intarray,delete_intarray,intarray_getitem,intarray_setitem,doublearray_getitem,doublearray_setitem,doublearray_cast,doublearray_frompointer,xyarray_getitem,xyarray_setitem,xyarray_cast,xyarray_frompointer,delete_abarray,abarray_getitem,abarray_setitem,shortarray_getitem,shortarray_setitem,shortarray_cast,shortarray_frompointer,sum_array));

// Check classes.
// NB An "li_carrays_cpp" class is created as a mock namespace.
check::classes(array(li_carrays_cpp,doubleArray,AB,XY,XYArray,shortArray));

// Check global variables.
check::globals(array(xy_x,xy_y,globalxyarray,ab_a,ab_b,globalabarray));

$d = new doubleArray(10);

$d->setitem(0, 7);
$d->setitem(5, $d->getitem(0) + 3);
check::equal($d->getitem(0) + $d->getitem(5), 17., "7+10==17");

check::done();
?>
