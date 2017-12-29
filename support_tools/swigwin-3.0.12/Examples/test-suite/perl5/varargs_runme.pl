#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 7;
BEGIN { use_ok('varargs') }
require_ok('varargs');

is(varargs::test("Hello"), "Hello");

my $f = new varargs::Foo("BuonGiorno", 1);
is($f->{str}, "BuonGiorno");

$f = new varargs::Foo("Greetings");
is($f->{str}, "Greetings");
        
is($f->test("Hello"), "Hello");

is(varargs::Foo::statictest("Grussen", 1), "Grussen");
