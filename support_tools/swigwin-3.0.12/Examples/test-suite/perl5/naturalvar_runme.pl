#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 5;
BEGIN { use_ok('naturalvar') }
require_ok('naturalvar');

# adapted from ../python/naturalvar_runme.py

my $f = naturalvar::Foo->new();
isa_ok($f, 'naturalvar::Foo');
my $b = naturalvar::Bar->new();
isa_ok($b, 'naturalvar::Bar');

$b->{f} = $f;

$naturalvar::s = "hello";

$b->{s} = "hello";

is($naturalvar::s, $b->{s});

