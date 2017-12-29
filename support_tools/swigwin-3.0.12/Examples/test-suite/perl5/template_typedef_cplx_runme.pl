#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 14;
BEGIN { use_ok('template_typedef_cplx') }
require_ok('template_typedef_cplx');

# adapted from ../python/template_typedef_cplx_runme.py

{	# kids, don't try this at home (glob hijinks)
	my $cvar = *template_typedef_cplx::;
	map { ${*::}{$_} = ${$cvar}{$_} } keys %{$cvar};
}

#
# double case
#

my $d = eval { make_Identity_double() };
ok(ref($d), 'is an object');
like(ref($d), qr/ArithUnaryFunction/, "is an ArithUnaryFunction");

my $e = eval { make_Multiplies_double_double_double_double($d, $d) };
ok(ref($e), 'is an object');
like(ref($e), qr/ArithUnaryFunction/, "is an ArithUnaryFunction");

#
# complex case
#

my $c = eval { make_Identity_complex() };
ok(ref($c), 'is an object');
like(ref($c), qr/ArithUnaryFunction/, "is an ArithUnaryFunction");

my $f = eval { make_Multiplies_complex_complex_complex_complex($c, $c) };
ok(ref($f), 'is an object');
like(ref($f), qr/ArithUnaryFunction/, "is an ArithUnaryFunction");

#
# Mix case
#

my $g = eval { make_Multiplies_double_double_complex_complex($d, $c) };
ok(ref($f), 'is an object');
like(ref($f), qr/ArithUnaryFunction/, "is an ArithUnaryFunction");

my $h = eval { make_Multiplies_complex_complex_double_double($c, $d) };
ok(ref($h), 'is an object');
like(ref($h), qr/ArithUnaryFunction/, "is an ArithUnaryFunction");
