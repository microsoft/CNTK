<?php

require "tests.php";
require "grouping.php";

check::functions(array("test1","test2","do_unary","negate"));
check::equal(5,test1(5),"5==test1(5)");
check::resource(test2(7),"_p_int","_p_int==test2(7)");
check::globals(array(test3));

//check::equal(37,test3_get(),'37==test3_get()');
check::equal(37,check::get("test3"),'37==get(test3)');
//test3_set(38);
check::set(test3,38); 
//check::equal(38,test3_get(),'38==test3_get() after test3_set(37)');
check::equal(38,check::get(test3),'38==get(test3) after set(test)');

check::equal(-5,negate(5),"-5==negate(5)");
check::equal(7,do_unary(-7,NEGATE),"7=do_unary(-7,NEGATE)");

check::done();
?>
