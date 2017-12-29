#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 3;
BEGIN { use_ok('using1') }
require_ok('using1');

# adapted from ../python/using1_runme.py

is(using1::spam(37), 37);
