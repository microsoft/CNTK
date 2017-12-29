<?php

require "tests.php";
require "char_strings.php";

$CPLUSPLUS_MSG = "A message from the deep dark world of C++, where anything is possible.";
$OTHERLAND_MSG_10 = "Little message from the safe world.10";

check::equal(GetCharHeapString(), $CPLUSPLUS_MSG, "failed GetCharHeapString");
check::equal(GetConstCharProgramCodeString(), $CPLUSPLUS_MSG, "failed GetConstCharProgramCodeString");
check::equal(GetCharStaticString(), $CPLUSPLUS_MSG, "failed GetCharStaticString");
check::equal(GetCharStaticStringFixed(), $CPLUSPLUS_MSG, "failed GetCharStaticStringFixed");
check::equal(GetConstCharStaticStringFixed(), $CPLUSPLUS_MSG, "failed GetConstCharStaticStringFixed");

check::equal(SetCharHeapString($OTHERLAND_MSG_10, 10), true, "failed GetConstCharStaticStringFixed");
check::equal(SetCharStaticString($OTHERLAND_MSG_10, 10), true, "failed SetCharStaticString");
check::equal(SetCharArrayStaticString($OTHERLAND_MSG_10, 10), true, "failed SetCharArrayStaticString");
check::equal(SetConstCharHeapString($OTHERLAND_MSG_10, 10), true, "failed SetConstCharHeapString");
check::equal(SetConstCharStaticString($OTHERLAND_MSG_10, 10), true, "failed SetConstCharStaticString");
check::equal(SetConstCharArrayStaticString($OTHERLAND_MSG_10, 10), true, "failed SetConstCharArrayStaticString");

check::equal(CharPingPong($OTHERLAND_MSG_10), $OTHERLAND_MSG_10, "failed CharPingPong");

Global_char_set($OTHERLAND_MSG_10);
check::equal(Global_char_get(), $OTHERLAND_MSG_10, "failed Global_char_get");

Global_char_array1_set($OTHERLAND_MSG_10);
check::equal(Global_char_array1_get(), $OTHERLAND_MSG_10, "failed Global_char_array1_get");

Global_char_array2_set($OTHERLAND_MSG_10);
check::equal(Global_char_array2_get(), $OTHERLAND_MSG_10, "failed Global_char_array2_get");

check::equal(Global_const_char_get(), $CPLUSPLUS_MSG, "failed Global_const_char");
check::equal(Global_const_char_array1_get(), $CPLUSPLUS_MSG, "failed Global_const_char_array1");
check::equal(Global_const_char_array2_get(), $CPLUSPLUS_MSG, "failed Global_const_char_array2");

check::equal(GetCharPointerRef(), $CPLUSPLUS_MSG, "failed GetCharPointerRef");
check::equal(SetCharPointerRef($OTHERLAND_MSG_10, 10), true, "failed SetCharPointerRef");
check::equal(GetConstCharPointerRef(), $CPLUSPLUS_MSG, "failed GetConstCharPointerRef");
check::equal(SetConstCharPointerRef($OTHERLAND_MSG_10, 10), true, "failed SetConstCharPointerRef");

check::done();
?>
