#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 3;
BEGIN { use_ok('dynamic_cast') }
require_ok('dynamic_cast');

my $f = dynamic_cast::Foo->new();
my $b = dynamic_cast::Bar->new();
my $x = $f->blah();
my $y = $b->blah();
my $a = dynamic_cast::do_test($y);
is($a, "Bar::test");
