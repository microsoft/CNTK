<?php

require "tests.php";
require "template_arg_typename.php";

// No new functions
check::functions(array());
// No new classes
check::classes(array(UnaryFunction_bool_bool,BoolUnaryFunction_bool));
$ufbb=new unaryfunction_bool_bool();
check::is_a($ufbb,"unaryfunction_bool_bool");

unset($whatisthis);
$bufb=new boolunaryfunction_bool($whatisthis);
check::is_a($bufb,"boolunaryfunction_bool");

check::done();
?>
