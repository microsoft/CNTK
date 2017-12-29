<?php
// Sample test file

require "tests.php";
require "____.php";

// No new functions
check::functions(array());
// No new classes
check::classes(array());
// now new vars
check::globals(array());

check::done();
?>
