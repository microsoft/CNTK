<?php

require "tests.php";
require "arrayptr.php";

// No new functions
check::functions(array(foo));
// No new classes
check::classes(array());
// now new vars
check::globals(array());

check::done();
?>
