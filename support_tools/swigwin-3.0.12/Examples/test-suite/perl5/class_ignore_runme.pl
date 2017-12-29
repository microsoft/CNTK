#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 3;
BEGIN { use_ok('class_ignore') }
require_ok('class_ignore');

# adapted from ../python/class_ignore_runme.py

my $a = class_ignore::Bar->new();

is(class_ignore::do_blah($a), "Bar::blah");
