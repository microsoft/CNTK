<?php

require "tests.php";
require "cpp_basic.php";

// New functions
check::functions(array(foo_func1,foo_func2,foo___str__,foosubsub___str__,bar_test,bar_testfoo,get_func1_ptr,get_func2_ptr,test_func_ptr,fl_window_show));
// New classes
check::classes(array(cpp_basic,Foo,FooSub,FooSubSub,Bar,Fl_Window));
// New vars
check::globals(array(foo_num,foo_func_ptr,bar_fptr,bar_fref,bar_fval,bar_cint,bar_global_fptr,bar_global_fref,bar_global_fval));

$f = new Foo(3);
$f->func_ptr = get_func1_ptr();
check::equal(test_func_ptr($f, 7), 2*7*3, "get_func1_ptr() didn't work");
$f->func_ptr = get_func2_ptr();
check::equal(test_func_ptr($f, 7), -7*3, "get_func2_ptr() didn't work");

check::done();
?>
