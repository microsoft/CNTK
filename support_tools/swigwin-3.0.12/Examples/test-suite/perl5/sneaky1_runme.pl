#!/usr/bin/perl
use strict;
use warnings;
use Test::More 'no_plan';
BEGIN { use_ok('sneaky1') }
require_ok('sneaky1');

# adapted from ../python/sneaky1_runme.py

is(sneaky1::add(3,4), 7);
is(sneaky1::subtract(3,4), -1);
is(sneaky1::mul(3,4), 12);
is(sneaky1::divide(3,4), 0);
