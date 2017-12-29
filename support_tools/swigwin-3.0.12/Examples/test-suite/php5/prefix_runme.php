<?php

require "tests.php";
require "prefix.php";

// No new functions
check::functions(array(foo_get_self));
// No new classes
check::classes(array(ProjectFoo));
// now new vars
check::globals(array());

$f = new ProjectFoo();
// This resulted in "Fatal error: Class 'Foo' not found"
$f->get_self();

check::done();
?>
