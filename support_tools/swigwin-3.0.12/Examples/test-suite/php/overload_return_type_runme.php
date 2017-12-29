<?php

require "tests.php";
require "overload_return_type.php";

$b = new B;
check::equal($b->foo(1), 0, "");
check::classname("A", $b->foo("test"));

check::equal(overload_return_type::foo(), 1, "overload_return_type::foo() should be 1");
check::equal(overload_return_type::bar(), 1, "overload_return_type::bar() should be 1");

?>
