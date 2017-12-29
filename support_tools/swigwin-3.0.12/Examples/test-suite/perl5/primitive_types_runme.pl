use strict;
use warnings;
use Test::More tests => 51;
BEGIN { use_ok('primitive_types') }
require_ok('primitive_types');

primitive_types::var_init();

# assigning globals calls
$primitive_types::var_bool = $primitive_types::sct_bool;
$primitive_types::var_schar = $primitive_types::sct_schar;
$primitive_types::var_uchar = $primitive_types::sct_uchar;
$primitive_types::var_int = $primitive_types::sct_int;
$primitive_types::var_uint = $primitive_types::sct_uint;
$primitive_types::var_short = $primitive_types::sct_short;
$primitive_types::var_ushort = $primitive_types::sct_ushort;
$primitive_types::var_long = $primitive_types::sct_long;
$primitive_types::var_ulong = $primitive_types::sct_ulong;
$primitive_types::var_llong = $primitive_types::sct_llong;
$primitive_types::var_ullong = $primitive_types::sct_ullong;
$primitive_types::var_char = $primitive_types::sct_char;
$primitive_types::var_pchar = $primitive_types::sct_pchar;
$primitive_types::var_pcharc = $primitive_types::sct_pcharc;
$primitive_types::var_pint = $primitive_types::sct_pint;
$primitive_types::var_sizet = $primitive_types::sct_sizet;
$primitive_types::var_hello = $primitive_types::sct_hello;
$primitive_types::var_myint = $primitive_types::sct_myint;
$primitive_types::var_namet = $primitive_types::def_namet;
$primitive_types::var_parami = $primitive_types::sct_parami;
$primitive_types::var_paramd = $primitive_types::sct_paramd;
$primitive_types::var_paramc = $primitive_types::sct_paramc;

ok(primitive_types::v_check(), "v_check");

#def pyerror(name, val, cte):
#  print "bad val/cte", name, val, cte
#  raise RuntimeError
#  pass

is($primitive_types::var_bool, $primitive_types::cct_bool, "bool");
is($primitive_types::var_schar, $primitive_types::cct_schar, "schar");
is($primitive_types::var_uchar, $primitive_types::cct_uchar, "uchar");
is($primitive_types::var_int, $primitive_types::cct_int, "int");
is($primitive_types::var_uint, $primitive_types::cct_uint, "uint");
is($primitive_types::var_short, $primitive_types::cct_short, "short");
is($primitive_types::var_ushort, $primitive_types::cct_ushort, "ushort");
is($primitive_types::var_long, $primitive_types::cct_long, "long");
is($primitive_types::var_ulong, $primitive_types::cct_ulong, "ulong");
is($primitive_types::var_llong, $primitive_types::cct_llong, "llong");
is($primitive_types::var_ullong, $primitive_types::cct_ullong, "ullong");
is($primitive_types::var_char, $primitive_types::cct_char, "char");
is($primitive_types::var_pchar, $primitive_types::cct_pchar, "pchar");
is($primitive_types::var_pcharc, $primitive_types::cct_pcharc, "pchar");
is($primitive_types::var_pint, $primitive_types::cct_pint, "pint");
is($primitive_types::var_sizet, $primitive_types::cct_sizet, "sizet");
is($primitive_types::var_hello, $primitive_types::cct_hello, "hello");
is($primitive_types::var_myint, $primitive_types::cct_myint, "myint");
is($primitive_types::var_namet, $primitive_types::def_namet, "name");

#class PyTest (TestDirector):
#  def __init__(self):
#    TestDirector.__init__(self)
#    pass
#  def ident(self, x):
#    return x
#  
#  def vval_bool(self, x): return self.ident(x)
#  def vval_schar(self, x): return self.ident(x)
#  def vval_uchar(self, x): return self.ident(x)
#  def vval_int(self, x): return self.ident(x)
#  def vval_uint(self, x): return self.ident(x)
#  def vval_short(self, x): return self.ident(x)
#  def vval_ushort(self, x): return self.ident(x)
#  def vval_long(self, x): return self.ident(x)
#  def vval_ulong(self, x): return self.ident(x)
#  def vval_llong(self, x): return self.ident(x)
#  def vval_ullong(self, x): return self.ident(x)
#  def vval_float(self, x): return self.ident(x)
#  def vval_double(self, x): return self.ident(x)
#  def vval_char(self, x): return self.ident(x)
#  def vval_pchar(self, x): return self.ident(x)
#  def vval_pcharc(self, x): return self.ident(x)
#  def vval_pint(self, x): return self.ident(x)
#  def vval_sizet(self, x): return self.ident(x)
#  def vval_hello(self, x): return self.ident(x)
#  def vval_myint(self, x): return self.ident(x)
#
#  def vref_bool(self, x): return self.ident(x)
#  def vref_schar(self, x): return self.ident(x)
#  def vref_uchar(self, x): return self.ident(x)
#  def vref_int(self, x): return self.ident(x)
#  def vref_uint(self, x): return self.ident(x)
#  def vref_short(self, x): return self.ident(x)
#  def vref_ushort(self, x): return self.ident(x)
#  def vref_long(self, x): return self.ident(x)
#  def vref_ulong(self, x): return self.ident(x)
#  def vref_llong(self, x): return self.ident(x)
#  def vref_ullong(self, x): return self.ident(x)
#  def vref_float(self, x): return self.ident(x)
#  def vref_double(self, x): return self.ident(x)
#  def vref_char(self, x): return self.ident(x)
#  def vref_pchar(self, x): return self.ident(x)
#  def vref_pcharc(self, x): return self.ident(x)
#  def vref_pint(self, x): return self.ident(x)
#  def vref_sizet(self, x): return self.ident(x)
#  def vref_hello(self, x): return self.ident(x)
#  def vref_myint(self, x): return self.ident(x)
#
#  pass


my $t = primitive_types::Test->new();
#p = PyTest()
#
#
# internal call check
#if t.c_check() != p.c_check():
#  raise RuntimeError, "bad director"
#
#p.var_bool = p.stc_bool
#p.var_schar = p.stc_schar
#p.var_uchar = p.stc_uchar
#p.var_int = p.stc_int
#p.var_uint = p.stc_uint
#p.var_short = p.stc_short
#p.var_ushort = p.stc_ushort
#p.var_long = p.stc_long
#p.var_ulong = p.stc_ulong
#p.var_llong = p.stc_llong
#p.var_ullong = p.stc_ullong
#p.var_char = p.stc_char
#p.var_pchar = sct_pchar
#p.var_pcharc = sct_pcharc
#p.var_pint = sct_pint
#p.var_sizet = sct_sizet
#p.var_hello = sct_hello
#p.var_myint = sct_myint
#p.var_namet = def_namet
#p.var_parami = sct_parami
#p.var_paramd = sct_paramd
#p.var_paramc = sct_paramc
#
#p.v_check()

$t->{var_bool} = $primitive_types::Test::stc_bool;
$t->{var_schar} = $primitive_types::Test::stc_schar;
$t->{var_uchar} = $primitive_types::Test::stc_uchar;
$t->{var_int} = $primitive_types::Test::stc_int;
$t->{var_uint} = $primitive_types::Test::stc_uint;
$t->{var_short} = $primitive_types::Test::stc_short;
$t->{var_ushort} = $primitive_types::Test::stc_ushort;
$t->{var_long} = $primitive_types::Test::stc_long;
$t->{var_ulong} = $primitive_types::Test::stc_ulong;
$t->{var_llong} = $primitive_types::Test::stc_llong;
$t->{var_ullong} = $primitive_types::Test::stc_ullong;
$t->{var_char} = $primitive_types::Test::stc_char;
$t->{var_pchar} = $primitive_types::sct_pchar;
$t->{var_pcharc} = $primitive_types::sct_pcharc;
$t->{var_pint} = $primitive_types::sct_pint;
$t->{var_sizet} = $primitive_types::sct_sizet;
$t->{var_hello} = $primitive_types::sct_hello;
$t->{var_myint} = $primitive_types::sct_myint;
$t->{var_namet} = $primitive_types::def_namet;
$t->{var_parami} = $primitive_types::sct_parami;
$t->{var_paramd} = $primitive_types::sct_paramd;
$t->{var_paramc} = $primitive_types::sct_paramc;
ok($t->v_check(), 'v_check');

is($primitive_types::def_namet, "hola", "namet");
$t->{var_namet} = $primitive_types::def_namet;
is($t->{var_namet}, $primitive_types::def_namet, "namet");

$t->{var_namet} = 'hola';

is($t->{var_namet}, 'hola', "namet");

$t->{var_namet} = 'hol';

is($t->{var_namet}, 'hol', "namet");


$primitive_types::var_char = "\0";
is($primitive_types::var_char, "\0", "char '0' case");
  
$primitive_types::var_char = 0;
is($primitive_types::var_char, "\0", "char '0' case");

$primitive_types::var_namet = "\0";
is($primitive_types::var_namet, '', "char '\\0' case");

$primitive_types::var_namet = '';
is($primitive_types::var_namet, '', "char empty case");

$primitive_types::var_pchar = undef;
is($primitive_types::var_pchar, undef, "undef case");

$primitive_types::var_pchar = '';
is($primitive_types::var_pchar, '', "char empty case");

$primitive_types::var_pcharc = undef;
is($primitive_types::var_pcharc, undef, "undef case");

$primitive_types::var_pcharc = '';
is($primitive_types::var_pcharc, '', "char empty case");


#
# creating a raw char*
#
my $pc = primitive_types::new_pchar(5);
primitive_types::pchar_setitem($pc, 0, 'h');
primitive_types::pchar_setitem($pc, 1, 'o');
primitive_types::pchar_setitem($pc, 2, 'l');
primitive_types::pchar_setitem($pc, 3, 'a');
primitive_types::pchar_setitem($pc, 4, 0);


$primitive_types::var_pchar = $pc;
is($primitive_types::var_pchar, "hola", "pointer case");

$primitive_types::var_namet = $pc;
is($primitive_types::var_namet, "hola", "pointer case");

primitive_types::delete_pchar($pc);

#
# Now when things should fail
#

{
	my $orig = $t->{var_uchar};
	eval { $t->{var_uchar} = 10000 };
	like($@, qr/\bOverflowError\b/, "uchar typemap");
	is($orig, $t->{var_uchar}, "uchar typemap");
}
{
	my $orig = $t->{var_char};
	#eval { $t->{var_char} = "23" }; Perl will gladly make a number out of that
	eval { $t->{var_char} = "twenty-three" };
	like($@, qr/\bTypeError\b/, "char typemap");
	is($orig, $t->{var_char}, "char typemap");
}
{
	my $orig = $t->{var_uint};
	eval { $t->{var_uint} = -1 };
	like($@, qr/\bOverflowError\b/, "uint typemap");
	is($orig, $t->{var_uint}, "uint typemap");
}
{
	my $orig = $t->{var_namet};
	eval { $t->{var_namet} = '123456' };
	like($@, qr/\bTypeError\b/, "namet typemap");
	is($orig, $t->{var_namet}, "namet typemap");
}
#t2 = p.vtest(t)
#if t.var_namet !=  t2.var_namet:
#  raise RuntimeError, "bad SWIGTYPE* typemap"

is($primitive_types::fixsize, "ho\0la\0\0\0", "FIXSIZE typemap");

$primitive_types::fixsize = 'ho';
is($primitive_types::fixsize, "ho\0\0\0\0\0\0", "FIXSIZE typemap");


my $f = primitive_types::Foo->new(3);
my $f1 = primitive_types::fptr_val($f);
my $f2 = primitive_types::fptr_ref($f);
is($f1->{_a}, $f2->{_a}, "const ptr& typemap");
  

is(primitive_types::char_foo(1,3), 3, "int typemap");

is(primitive_types::char_foo(1,"hello"), "hello", "char* typemap");
  
is(primitive_types::SetPos(1,3), 4, "int typemap");
