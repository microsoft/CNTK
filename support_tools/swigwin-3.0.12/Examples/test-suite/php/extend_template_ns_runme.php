<?php

require "tests.php";
require "extend_template_ns.php";

check::classes(array("extend_template_ns","Foo_One"));
$foo=new Foo_One();
check::equal(2,$foo->test1(2),"test1");
check::equal(3,$foo->test2(3),"test2");

check::done();
?>
