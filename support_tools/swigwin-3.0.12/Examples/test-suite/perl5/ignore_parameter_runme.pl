#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 14;
BEGIN { use_ok('ignore_parameter') }
require_ok('ignore_parameter');

# adapted from ../java/ignore_parameter_runme.java

# Runtime test checking the %typemap(ignore) macro

# Compilation will ensure the number of arguments and type are correct.
# Then check the return value is the same as the value given to the ignored parameter.
is(ignore_parameter::jaguar(200, 0.0), "hello", "jaguar()");
is(ignore_parameter::lotus("fast", 0.0), 101, "lotus()");
is(ignore_parameter::tvr("fast", 200), 8.8, "tvr()");
is(ignore_parameter::ferrari(), 101, "ferrari()");

my $sc = new ignore_parameter::SportsCars();
is($sc->daimler(200, 0.0), "hello", "daimler()");
is($sc->astonmartin("fast", 0.0), 101, "astonmartin()");
is($sc->bugatti("fast", 200), 8.8, "bugatti()");
is($sc->lamborghini(), 101, "lamborghini()");

# Check constructors are also generated correctly
my $mc = eval { new ignore_parameter::MiniCooper(200, 0.0) };
isa_ok($mc, 'ignore_parameter::MiniCooper');
my $mm = eval { new ignore_parameter::MorrisMinor("slow", 0.0) };
isa_ok($mm, 'ignore_parameter::MorrisMinor');
my $fa = eval { new ignore_parameter::FordAnglia("slow", 200) };
isa_ok($fa, 'ignore_parameter::FordAnglia');
my $aa = eval { new ignore_parameter::AustinAllegro() };
isa_ok($aa, 'ignore_parameter::AustinAllegro');
