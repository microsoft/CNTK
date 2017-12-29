# file: runme.pl

# This file test multiple inheritance

use example;

$foo_Bar = new example::Foo_Bar();

print "must be foo: ";
$foo_Bar->foo();

print "must be bar: ";
$foo_Bar->bar();

print "must be foobar: ";
$foo_Bar->fooBar();
