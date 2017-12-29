<?php

require "example.php";

# ----- Object creation -----

# Print out the value of some enums
print "*** color ***";
print "    RED    =" . RED;
print "    BLUE   =" . BLUE;
print "    GREEN  =" . GREEN;

print "\n*** Foo::speed ***";
print "    Foo_IMPULSE   =" . Foo_IMPULSE;
print "    Foo_WARP      =" . Foo_WARP;
print "    Foo_LUDICROUS =" . Foo_LUDICROUS;

print "\nTesting use of enums with functions\n";

enum_test(RED, Foo_IMPULSE);
enum_test(BLUE, Foo_WARP);
enum_test(GREEN, Foo_LUDICROUS);
enum_test(1234,5678);

print "\nTesting use of enum with class method\n";
$f = new_Foo();

Foo_enum_test($f,Foo_IMPULSE);
Foo_enum_test($f,Foo_WARP);
Foo_enum_test($f,Foo_LUDICROUS);

?>
