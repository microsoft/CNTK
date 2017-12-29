<?php

require "tests.php";
require "newobject1.php";

// No new functions
check::functions(array(foo_makefoo,foo_makemore,foo_foocount));
// No new classes
check::classes(array(Foo));
// now new vars
check::globals(array());

$foo = Foo::makeFoo();
check::equal(get_class($foo), "Foo", "static failed");
$bar = $foo->makeMore();
check::equal(get_class($bar), "Foo", "regular failed");

check::done();
?>
