#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 6;
BEGIN { use_ok('preproc') }
require_ok('preproc');

# adapted from ../python/preproc_runme.py

is($preproc::endif, 1);
is($preproc::define, 1);
is($preproc::defined, 1);
is($preproc::one * 2, $preproc::two);

