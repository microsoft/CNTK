<?php

require "tests.php";
require "extend_template.php";

check::classes(array("Foo_0"));
$foo=new Foo_0();
check::equal(2,$foo->test1(2),"test1");
check::equal(3,$foo->test2(3),"test2");

check::done();
?>
