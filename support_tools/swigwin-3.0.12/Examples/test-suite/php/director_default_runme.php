<?php

require "tests.php";
require "director_default.php";

// No new functions
check::functions(array(foo_msg,foo_getmsg,bar_msg,bar_getmsg,defaultsbase_defaultargs,defaultsderived_defaultargs));
// No new classes
check::classes(array(Foo,Bar,DefaultsBase,DefaultsDerived));
// now new vars
check::globals(array());

$f = new Foo();
$f = new Foo(1);

$f = new Bar();
$f = new Bar(1);

check::done();
?>
