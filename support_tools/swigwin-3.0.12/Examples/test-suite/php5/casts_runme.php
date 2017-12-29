<?php

require "tests.php";
require "casts.php";

// No new functions
check::functions(array(new_a,a_hello,new_b));
// No new classes
check::classes(array(A,B));
// now new vars
check::globals(array());

# Make sure $b inherites hello() from class A
$b=new B();
$b->hello();

check::done();
?>
