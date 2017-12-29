#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 16;
BEGIN { use_ok('template_typedef_cplx3') }
require_ok('template_typedef_cplx3');

# adapted from ../python/template_typedef_cplx3_runme.py

{       # kids, don't try this at home (glob hijinks)
        my $cvar = *template_typedef_cplx3::;
        map { ${*::}{$_} = ${$cvar}{$_} } keys %{$cvar};
}

my $s = Sin->new();
is($s->get_base_value(), 0);
is($s->get_value(), 1);
is($s->get_arith_value(), 2);
is(my_func_r($s), 0);
isa_ok(make_Multiplies_double_double_double_double($s,$s),
  "template_typedef_cplx3::ArithUnaryFunction_double_double");

my $z = CSin->new();
is($z->get_base_value(), 0);
is($z->get_value(), 1);
is($z->get_arith_value(), 2);
is(my_func_c($z), 1);
isa_ok(make_Multiplies_complex_complex_complex_complex($z,$z),
  "template_typedef_cplx3::ArithUnaryFunction_complex_complex");

my $d = eval { make_Identity_double() };
isa_ok($d, "template_typedef_cplx3::ArithUnaryFunction_double_double");
is(my_func_r($d), 0);

my $c = eval { make_Identity_complex() };
isa_ok($d, "template_typedef_cplx3::ArithUnaryFunction_double_double");
is(my_func_c($c), 1);
  



