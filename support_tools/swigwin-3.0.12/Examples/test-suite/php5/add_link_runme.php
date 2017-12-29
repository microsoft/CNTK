<?php

require "tests.php";
require "add_link.php";

// No new functions, except the flat functions
check::functions(array(new_foo,foo_blah));

check::classes(array(Foo));

$foo=new foo();
check::is_a($foo,foo);

$foo_blah=$foo->blah();
check::is_a($foo_blah,foo);

//fails, can't be called as a class method, should allow and make it nil?
//$class_foo_blah=foo::blah();
//check::is_a($class_foo_blah,foo);

check::done();
?>
