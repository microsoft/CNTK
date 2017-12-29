#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 4;
BEGIN { use_ok('return_const_value') }
require_ok('return_const_value');

# adapted from ../python/return_const_value_runme.py

is(return_const_value::Foo_ptr::getPtr()->getVal(), 17);

is(return_const_value::Foo_ptr::getConstPtr()->getVal(), 17);

