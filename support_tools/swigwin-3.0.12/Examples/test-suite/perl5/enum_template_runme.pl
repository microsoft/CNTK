#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 4;
BEGIN { use_ok('enum_template') }
require_ok('enum_template');

# adapted from ../python/enum_template_runme.py

is(enum_template::MakeETest(), 1);

is(enum_template::TakeETest(0), undef);
