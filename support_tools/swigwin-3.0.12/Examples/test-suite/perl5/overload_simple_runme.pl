#!/usr/bin/perl -w
use overload_simple;
use vars qw/$DOWARN/;
use strict;
use Test::More tests => 97;

pass("loaded");

my $f = new overload_simple::Foo();
isa_ok($f, "overload_simple::Foo");
my $b = new overload_simple::Bar();
isa_ok($b, "overload_simple::Bar");
my $v = overload_simple::malloc_void(32);
isa_ok($v, "_p_void");


#
# Silence warnings about bad types
#
BEGIN { $SIG{'__WARN__'} = sub { warn $_[0] if $DOWARN } }
#
#these tests should 'fail'
#
eval { overload_simple::fint("l") };
ok($@, "fint(int) - int");

eval { overload_simple::fint("3.5") };
ok($@, "fint(int) - double");

eval { overload_simple::fdouble("l") };
ok($@, "fint(double) - int");

eval { overload_simple::fdouble("1.5/2.0") };
ok($@, "fint(double) - double");

#
#enable the warnings again
#
$DOWARN =1;

#
# 'simple' dispatch (no overload) of int and double arguments
#

is(overload_simple::fint(3), "fint:int", "fint(int) - int");

is(overload_simple::fint("1"), "fint:int", "fint(int) - string int");

is(overload_simple::fint(3.0), "fint:int", "fint(int) - double");

is(overload_simple::fint("3.0"), "fint:int", "fint(int) - string double");

# old bad case that now works
my $n = 3;
$n = $n + 1;
is(overload_simple::fint($n), "fint:int", "fint(int) - int var");

is(overload_simple::fint(4/2), "fint:int", "fint(int) - divide int denom");

is(overload_simple::fint(4/2.0), "fint:int", "fint(int) - divide double denom");

is(overload_simple::fdouble(3), "fdouble:double", "fdouble(double) - int");

is(overload_simple::fdouble("3"), "fdouble:double", "fdouble(double) - string int");

is(overload_simple::fdouble(3.0), "fdouble:double", "fdouble(double) - double");

is(overload_simple::fdouble("3.0"), "fdouble:double", "fdouble(double) - string doubl");

#
# Overload between int and double
#
is(overload_simple::num(3), "num:int", "num(int) - int");

is(overload_simple::num("3"), "num:int", "num(int) - string int");

is(overload_simple::num(3.0), "num:double", "num(int) - double");

is(overload_simple::num("3.0"), "num:double", "num(int) - string double");

#
# Overload between int, double, char * and many types.
#
is(overload_simple::foo(3), "foo:int", "foo:int - int");

is(overload_simple::foo(3.0), "foo:double", "foo:double - double");

is(overload_simple::foo("3"), "foo:char *", "foo:char * - string int");

is(overload_simple::foo("3.0"), "foo:char *", "foo:char * - string double");

is(overload_simple::foo("hello"), "foo:char *", "foo:char * string");

is(overload_simple::foo($f), "foo:Foo *", "foo:Foo *");

is(overload_simple::foo($b), "foo:Bar *", "foo:Bar *");

is(overload_simple::foo($v), "foo:void *", "foo:void *");

is(overload_simple::blah(3), "blah:double", "blah:double");

is(overload_simple::blah("hello"), "blah:char *", "blah:char *");

my $s = new overload_simple::Spam();

is($s->foo(3), "foo:int", "Spam::foo:int");

is($s->foo(3.0), "foo:double", "Spam::foo(double)");

is($s->foo("hello"), "foo:char *", "Spam::foo:char *");

is($s->foo($f), "foo:Foo *", "Spam::foo(Foo *)");

is($s->foo($b), "foo:Bar *", "Spam::foo(Bar *)");

is($s->foo($v), "foo:void *", "Spam::foo(void *)");

is(overload_simple::Spam::bar(3), "bar:int", "Spam::bar(int)");

is(overload_simple::Spam::bar(3.0), "bar:double", "Spam::bar(double)");

is(overload_simple::Spam::bar("hello"), "bar:char *", "Spam::bar(char *)");

is(overload_simple::Spam::bar($f), "bar:Foo *", "Spam::bar(Foo *)");

is(overload_simple::Spam::bar($b), "bar:Bar *", "Spam::bar(Bar *)");

is(overload_simple::Spam::bar($v), "bar:void *", "Spam::bar(void *)");

# Test constructors

$s = new overload_simple::Spam();
isa_ok($s, "overload_simple::Spam");

is($s->{type}, "none", "Spam()");

$s = new overload_simple::Spam(3);
isa_ok($s, "overload_simple::Spam");

is($s->{type}, "int", "Spam(int)");

$s = new overload_simple::Spam(3.0);
isa_ok($s, "overload_simple::Spam");
is($s->{type}, "double", "Spam(double)");

$s = new overload_simple::Spam("hello");
isa_ok($s, "overload_simple::Spam");
is($s->{type}, "char *", "Spam(char *)");

$s = new overload_simple::Spam($f);
isa_ok($s, "overload_simple::Spam");
is($s->{type}, "Foo *", "Spam(Foo *)");

$s = new overload_simple::Spam($b);
isa_ok($s, "overload_simple::Spam");
is($s->{type}, "Bar *", "Spam(Bar *)");

$s = new overload_simple::Spam($v);
isa_ok($s, "overload_simple::Spam");
is($s->{type}, "void *", "Spam(void *)");

#
# Combine dispatch
#


is(overload_simple::fid(3, 3.0), "fid:intdouble", "fid(int,double)");

is(overload_simple::fid(3.0, 3), "fid:doubleint", "fid(double,int)");

is(overload_simple::fid(3.0, 3.0), "fid:doubledouble", "fid(double,double)");

is(overload_simple::fid(3, 3), "fid:intint", "fid(int,int)");

# with strings now

is(overload_simple::fid(3, "3.0"), "fid:intdouble", "fid(int,double)");

is(overload_simple::fid("3", 3.0), "fid:intdouble", "fid(int,double)");

is(overload_simple::fid("3", "3.0"), "fid:intdouble", "fid(int,double)");

is(overload_simple::fid(3.0, "3"), "fid:doubleint", "fid(double,int)");

is(overload_simple::fid("3.0", "3.0"), "fid:doubledouble", "fid:doubledouble");

is(overload_simple::fid("3", 3), "fid:intint", "fid:fid(int,int)");

isnt(overload_simple::fbool(0), overload_simple::fbool(1), "fbool(bool)");

is(2, overload_simple::fbool(2), "fbool(int)");

# int and object overload

is(overload_simple::int_object(1), 1, "int_object(1)");
is(overload_simple::int_object(0), 0, "int_object(0)");
is(overload_simple::int_object(undef), 999, "int_object(Spam*)");
is(overload_simple::int_object($s), 999, "int_object(Spam*)");

# some of this section is duplication of above tests, but I want to see
# parity with the coverage in wrapmacro_runme.pl.

sub check {
  my($args, $want) = @_;
  my($s, $rslt) = defined $want ?  ($want, "bar:$want") : ('*boom*', undef);
  is(eval("overload_simple::Spam::bar($args)"), $rslt, "bar($args) => $s");
}

# normal use patterns
check("11", 'int');
check("11.0", 'double');
check("'11'", 'char *');
check("'11.0'", 'char *');
check("-13", 'int');
check("-13.0", 'double');
check("'-13'", 'char *');
check("'-13.0'", 'char *');

check("' '", 'char *');
check("' 11 '", 'char *');
# TypeError explosions
check("\\*STDIN", undef);
check("[]", undef);
check("{}", undef);
check("sub {}", undef);

# regression cases
check("''", 'char *');
check("' 11'", 'char *');
check("' 11.0'", 'char *');
check("' -11.0'", 'char *');
check("\"11\x{0}\"", 'char *');
check("\"\x{0}\"", 'char *');
check("\"\x{9}11\x{0}this is not eleven.\"", 'char *');
check("\"\x{9}11.0\x{0}this is also not eleven.\"", 'char *');
