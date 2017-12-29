<?php

require "tests.php";
require "li_factory.php";

// No new functions
check::functions(array(geometry_draw,geometry_create,geometry_clone_,point_draw,point_width,point_clone_,circle_draw,circle_radius,circle_clone_));
// No new classes
check::classes(array(Geometry,Point,Circle));
// now new vars
check::globals(array());

$circle = Geometry::create(Geometry::CIRCLE);
$r = $circle->radius();
check::equal($r, 1.5, "r failed");

$point = Geometry::create(Geometry::POINT);
$w = $point->width();
check::equal($w, 1.0, "w failed");

check::done();
?>
