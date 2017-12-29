#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 6;
BEGIN { use_ok('grouping') }
require_ok('grouping');

# adapted from ../python/grouping_runme.py

is(grouping::test1(42), 42);

isnt(eval { grouping::test2(42) }, undef);

is(grouping::do_unary(37, $grouping::NEGATE), -37);

$grouping::test3 = 42;
is($grouping::test3, 42);
