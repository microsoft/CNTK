#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 6;
BEGIN { use_ok('typedef_class') }
require_ok('typedef_class');

# adapted from ../python/typedef_class_runme.py

my $a = typedef_class::RealA->new();
isa_ok($a, 'typedef_class::RealA');
$a->{a} = 3;
is($a->{a}, 3);
my $b = typedef_class::B->new();
isa_ok($b, 'typedef_class::B');
is($b->testA($a), 3);
