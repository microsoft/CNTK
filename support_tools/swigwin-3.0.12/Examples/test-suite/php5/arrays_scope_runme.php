<?php

require "tests.php";
require "arrays_scope.php";

// New functions
check::functions(array(new_bar,bar_blah));
// New classes
check::classes(array(arrays_scope,Bar));
// New vars
check::globals(array(bar_adata,bar_bdata,bar_cdata));

$bar=new bar();

check::done();
?>
