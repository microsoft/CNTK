<?php

require "tests.php";
require "typedef_reference.php";

check::functions(array(somefunc,otherfunc,new_intp,copy_intp,delete_intp,intp_assign,intp_value));
$int2=copy_intp(2);
check::equal(2,somefunc($int2)," test passing intp to somefunc");
$int3=copy_intp(3);
check::equal(3,otherfunc($int3)," test passing intp to otherfunc");

check::done();
?>
