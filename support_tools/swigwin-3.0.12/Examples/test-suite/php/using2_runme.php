<?php

require "tests.php";
require "using2.php";
if (! class_exists("_fooimpl")) die("_fooimpl class not found\n");
if (! 3==spam(3)) die("spam function not working right\n");

check::done();
?>
