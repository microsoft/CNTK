<?php

require "tests.php";
require "evil_diamond_ns.php";

check::classes(array("evil_diamond_ns","foo","bar","baz","spam"));
check::functions("test");
check::is_a("bar","foo");
check::is_a("baz","foo");
check::is_a("spam","foo");
check::is_a("spam","bar");
//No multiple inheritance
//check::is_a("spam","baz");
$spam=new spam();
$_spam=test($spam);

check::done();
?>
