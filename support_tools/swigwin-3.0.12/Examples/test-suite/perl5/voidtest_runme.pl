use strict;
use warnings;
use Test::More tests => 7;
BEGIN { use_ok('voidtest') }
require_ok('voidtest');

# adapted from ../python/voidtest_runme.py
voidtest::globalfunc();
my $f = voidtest::Foo->new();
is($f->memberfunc(), undef);
{ local $TODO = "opaque pointers hidden behind layer of indirection";
my $v1 = voidtest::vfunc1($f);
my $v2 = voidtest::vfunc2($f);
is($v1, $v2);
my $v3 = voidtest::vfunc3($v1);
is($v3->this, $f->this);
my $v4 = voidtest::vfunc4($f);
is($v1, $v4);
}
ok(1, "done");
