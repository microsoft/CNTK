<?php

require "tests.php";
require "conversion_namespace.php";

check::classes(array("Foo","Bar"));
$bar=new Bar;
check::classname("bar",$bar);
$foo=$bar->toFoo();
check::classname("foo",$foo);

check::done();
?>
