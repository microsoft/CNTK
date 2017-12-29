<?php

require "tests.php";
require "template_construct.php";

check::classes(array(Foo_int));
$foo_int=new foo_int(3);
check::is_a($foo_int,"foo_int","Made a foo_int");

check::done();
?>
