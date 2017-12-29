#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 19;
BEGIN { use_ok('reference_global_vars') }
require_ok('reference_global_vars');

# adapted from ../python/reference_global_vars_runme.py

my $cvar;
{
	# don't try this at home kids... sneaking an import of all symbols
	# from reference_global_vars to main because my fingers are getting
	# sore from qualifying all these names. ;)
	my $cvar = *reference_global_vars::;
	map { ${*::}{$_} = ${$cvar}{$_} } keys %{$cvar};
}

is(getconstTC()->{num}, 33);

# primitive reference variables
$cvar->{var_bool} = createref_bool(0);
is(value_bool($cvar->{var_bool}), '');

$cvar->{var_bool} = createref_bool(1);
is(value_bool($cvar->{var_bool}), 1);

$cvar->{var_char} = createref_char('w');
is(value_char($cvar->{var_char}), 'w');

$cvar->{var_unsigned_char} = createref_unsigned_char(10);
is(value_unsigned_char($cvar->{var_unsigned_char}), 10);

$cvar->{var_signed_char} = createref_signed_char(10);
is(value_signed_char($cvar->{var_signed_char}), 10);

$cvar->{var_short} = createref_short(10);
is(value_short($cvar->{var_short}), 10);

$cvar->{var_unsigned_short} = createref_unsigned_short(10);
is(value_unsigned_short($cvar->{var_unsigned_short}), 10);

$cvar->{var_int} = createref_int(10);
is(value_int($cvar->{var_int}), 10);

$cvar->{var_unsigned_int} = createref_unsigned_int(10);
is(value_unsigned_int($cvar->{var_unsigned_int}), 10);

$cvar->{var_long} = createref_long(10);
is(value_long($cvar->{var_long}), 10);

$cvar->{var_unsigned_long} = createref_unsigned_long(10);
is(value_unsigned_long($cvar->{var_unsigned_long}), 10);

SKIP: {
	use Math::BigInt qw();
	skip "64 bit int support", 1 unless eval { pack 'q', 1 };
	# the pack dance is to get plain old IVs out of the
	# Math::BigInt objects.
	my $a = unpack 'q', pack 'q', Math::BigInt->new('8070450532247928824');
	$cvar->{var_long_long} = createref_long_long($a);
	is(value_long_long($cvar->{var_long_long}), $a);
}

#ull = abs(0xFFFFFFF2FFFFFFF0)
my $ull = 55834574864;
$cvar->{var_unsigned_long_long} = createref_unsigned_long_long($ull);
is(value_unsigned_long_long($cvar->{var_unsigned_long_long}), $ull);

$cvar->{var_float} = createref_float(10.5);
is(value_float($cvar->{var_float}), 10.5);

$cvar->{var_double} = createref_double(10.5);
is(value_double($cvar->{var_double}), 10.5);

# class reference variable
$cvar->{var_TestClass} = createref_TestClass(
	TestClass->new(20)
);
is(value_TestClass($cvar->{var_TestClass})->{num}, 20);

