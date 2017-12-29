<?php

require "tests.php";
require "li_std_string.php";

// Global variables
//$s="initial string";
//check::equal(GlobalString2_get() ,"global string 2", "GlobalString2 test 1");

// Global variables
$s = "initial string";
check::equal(GlobalString2_get(), "global string 2", "GlobalString2 test 1");
GlobalString2_set($s);
check::equal(GlobalString2_get(), $s, "GlobalString2 test 2");
check::equal(ConstGlobalString_get(), "const global string", "ConstGlobalString test");

// Member variables
$myStructure = new Structure();
check::equal($myStructure->MemberString2, "member string 2", "MemberString2 test 1");
$myStructure->MemberString2 = $s;
check::equal($myStructure->MemberString2, $s, "MemberString2 test 2");
check::equal($myStructure->ConstMemberString, "const member string", "ConstMemberString test");

check::equal(Structure::StaticMemberString2(), "static member string 2", "StaticMemberString2 test 1");
Structure::StaticMemberString2($s);
check::equal(Structure::StaticMemberString2(), $s, "StaticMemberString2 test 2");
// below broken ?
//check::equal(Structure::ConstStaticMemberString(), "const static member string", "ConstStaticMemberString test");

check::done();
?>
