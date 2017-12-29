<?php

require "tests.php";
require "overload_rename.php";

// No new functions
check::functions(array());
// No new classes
check::classes(array(Foo));
// now new vars
check::globals(array());

$f = new Foo(1.0);
$f = new Foo(1.0,1.0);
$f = Foo::Foo_int(1.0,1);
$f = Foo::Foo_int(1.0,1,1.0);

check::done();
?>
