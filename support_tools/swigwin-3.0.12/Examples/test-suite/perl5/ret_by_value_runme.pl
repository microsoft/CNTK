#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 5;
BEGIN { use_ok('ret_by_value') }
require_ok('ret_by_value');

my $tst = ret_by_value::get_test();
isa_ok($tst, 'ret_by_value::test');
is($tst->{myInt}, 100);
is($tst->{myShort}, 200);

