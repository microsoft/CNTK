<?php

require "tests.php";
require "argout.php";

check::functions(array(incp,incr,inctr,new_intp,copy_intp,delete_intp,intp_assign,intp_value,voidhandle,handle));

$ip=copy_intp(42);
check::equal(42,incp($ip),"42==incp($ip)");
check::equal(43,intp_value($ip),"43=$ip");

$p=copy_intp(2);
check::equal(2,incp($p),"2==incp($p)");
check::equal(3,intp_value($p),"3==$p");

$r=copy_intp(7);
check::equal(7,incr($r),"7==incr($r)");
check::equal(8,intp_value($r),"8==$r");

$tr=copy_intp(4);
check::equal(4,inctr($tr),"4==incr($tr)");
check::equal(5,intp_value($tr),"5==$tr");

# Check the voidhandle call, first with null
unset($handle);
# FIXME: Call-time pass-by-reference has been deprecated for ages, and was
# removed in PHP 5.4.  We need to rework 
#voidhandle(&$handle);
#check::resource($handle,"_p_void",'$handle is not _p_void');
#$handledata=handle($handle);
#check::equal($handledata,"Here it is","\$handledata != \"Here it is\"");

unset($handle);
voidhandle($handle);
check::isnull($handle,'$handle not null');

check::done();
?>
