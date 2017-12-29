#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 3;
BEGIN { use_ok('using2') }
require_ok('using2');

# adapted from ../python/using2_runme.py

is(using2::spam(37), 37);
