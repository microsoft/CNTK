use strict;
use warnings;
use Test::More tests => 5;
BEGIN { use_ok('multiple_inheritance') }
require_ok('multiple_inheritance');

my $fooBar = new multiple_inheritance::FooBar();
is($fooBar->foo(), 2, "Runtime test1");

is($fooBar->bar(), 1, "Runtime test2");

is($fooBar->fooBar(), 3, "Runtime test3 ");
