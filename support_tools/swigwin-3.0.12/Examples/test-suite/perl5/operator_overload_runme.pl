#!/usr/bin/perl -w
use strict;
use Test::More tests => 39;

use operator_overload;

# Workaround for 
#   ok( not (expression) , "test description" );
# does not working in older versions of Perl, eg 5.004_04
sub ok_not ($;$) {
    my($test, $name) = @_;
    $test = not $test;
    ok($test, $name);
}

pass("loaded");

# first check all the operators are implemented correctly from pure C++ code
operator_overloadc::Op_sanity_check();

my $op = operator_overload::Op->new();

isa_ok($op, "operator_overload::Op");

my $op2 = operator_overload::Op->new();

isa_ok($op2, "operator_overload::Op");

# operator equal
$op->{i} = 5;
$op2->{i} = 3;

ok_not(($op == $op2), "operator equal: not equal");

$op->{i} = 3;
ok(($op == $op2), "operator equal: equal");

# operator not equal
$op->{i} = 5;
$op2->{i} = 3;

ok(($op != $op2), "operator not equal: not equal");

$op->{i} = 3;
ok_not(($op != $op2), "operator not equal: equal");

# stringify operator
$op->{i} = 3;
is("Op(3)", "$op", "operator stringify");

# addition operator
$op->{i} = 3;
$op2->{i} = 3;
my $op3 = $op + $op2;
is($op3->{i}, 6, "operator addition");

# addition assignment operator
$op->{i} = 3;
$op2->{i} = 3;
$op += $op2;
is($op->{i}, 6, "operator additive assignment");

# subtraction operator
$op3->{i} = 6;
$op2->{i} = 3;
$op = $op3 - $op2;
is($op->{i}, 3, "operator subtraction");

# reversed subtraction operator (with int)
$op3->{i} = 3;
$op = 6 - $op3;
is($op->{i}, 3, "reversed operator subtraction (with int)");

# subtractive assignment operator
$op->{i} = 6;
$op2->{i} = 3;
$op -= $op2;
is($op->{i}, 3, "operator subtractive assignment");

# multiplication operator
$op->{i} = 3;
$op2->{i} = 3;
$op3 = $op * $op2;
is($op3->{i}, 9, "operator multiplication");

# division operator
$op->{i} = 9;
$op2->{i} = 3;
$op3 = $op / $op2;
is($op3->{i}, 3, "operator division");

# modulus operator
$op->{i} = 8;
$op2->{i} = 3;
$op3 = $op % $op2;
is($op3->{i}, 2, "operator modulus");

# greater than operator
$op->{i} = 8;
$op2->{i} = 3;
ok($op > $op2, "operator greater than");
ok_not(($op2 > $op), "operator greater than");
$op->{i} = 3;
ok_not(($op2 > $op), "operator greater than");
ok_not(($op > $op2), "operator greater than");

# greater than or equal operator
$op->{i} = 8;
$op2->{i} = 3;
ok($op >= $op2, "operator greater than or equal");
ok_not(($op2 >= $op), "operator greater than or equal");
$op->{i} = 3;
ok(($op2 >= $op), "operator greater than or equal");
ok(($op >= $op2), "operator greater than or equal");

# lesser than operator
$op2->{i} = 8;
$op->{i} = 3;
ok($op < $op2, "operator lesser than");
ok_not(($op2 < $op), "operator lesser than");
$op2->{i} = 3;
ok_not(($op2 < $op), "operator lesser than");
ok_not(($op < $op2), "operator lesser than");

# less than or equal operator
$op2->{i} = 8;
$op->{i} = 3;
ok($op <= $op2, "operator lesser than or equal");
ok_not(($op2 <= $op), "operator lesser than or equal");
$op2->{i} = 3;
ok(($op2 <= $op), "operator less than or equal");
ok(($op <= $op2), "operator less than or equal");

# post-increment operator
$op->{i} = 7;
$op++;
is($op->{i}, 8, "operator post-increment");

# pre-increment operator
$op->{i} = 7;
++$op;
is($op->{i}, 8, "operator pre-increment");

# post-decrement operator
$op->{i} = 7;
$op--;
is($op->{i}, 6, "operator post-decrement");

# pre-decrement operator
$op->{i} = 7;
--$op;
is($op->{i}, 6, "operator pre-decrement");

# neg operator
$op->{i} = 3;
$op2 = -$op;
is($op2->{i}, -3, "operator neg");

# not operator
$op->{i} = 0;
is(!$op, !0, "operator not");

$op->{i} = 1;
is(!$op, !1, "operator not");

### # and operator
### $op->{i} = 4;
### $op2->{i} = 2;
### 
### is($op & $op2, 4 & 2, "operator and");
### 
### isnt(($op & $op2), (10 & 2), "operator and - false");

# fail("testing failed condition");

