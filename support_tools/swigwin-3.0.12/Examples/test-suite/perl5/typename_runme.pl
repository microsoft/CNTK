#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 4;
BEGIN { use_ok('typename') }
require_ok('typename');

# adapted from ../python/typename_runme.py

my $f = typename::Foo->new();
my $b = typename::Bar->new();
my $x = typename::twoFoo($f);
is($x, 4.3656);
my $y = typename::twoBar($b);
is($y, 84);
# I would like this test better if I could pass in a float to the
# integer test and see it lose precision.
