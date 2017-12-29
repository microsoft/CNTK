use strict;
use warnings;
use Test::More tests => 40;
BEGIN { use_ok('default_args') }
require_ok('default_args');

my $true = 1;
my $false = '';

is(default_args::anonymous(), 7771, "anonymous (1)");
is(default_args::anonymous(1234), 1234, "anonymous (2)");

is(default_args::booltest(), $true, "booltest (1)");
is(default_args::booltest($true), $true, "booltest (2)");
is(default_args::booltest($false), $false, "booltest (3)");

my $ec = new default_args::EnumClass();
is($ec->blah(), $true, "EnumClass");

is(default_args::casts1(), undef, "casts1");
is(default_args::casts2(), "Hello", "casts2");
is(default_args::casts1("Ciao"), "Ciao", "casts1 not default");
is(default_args::chartest1(), 'x', "chartest1");
is(default_args::chartest2(), "\0", "chartest2");
is(default_args::chartest1('y'), 'y', "chartest1 not default");
is(default_args::reftest1(), 42, "reftest1");
is(default_args::reftest1(400), 400, "reftest1 not default");
is(default_args::reftest2(), "hello", "reftest2");

# rename
my $foo = new default_args::Foo();
can_ok($foo, qw(newname renamed3arg renamed2arg renamed1arg));
eval {
	$foo->newname(); 
	$foo->newname(10); 
	$foo->renamed3arg(10, 10.0); 
	$foo->renamed2arg(10); 
	$foo->renamed1arg();
};
ok(not($@), '%rename handling');
 
# exception specifications
eval { default_args::exceptionspec() };
like($@, qr/^ciao/, "exceptionspec 1");
eval { default_args::exceptionspec(-1) };
like($@, qr/^ciao/, "exceptionspec 2");
eval { default_args::exceptionspec(100) };
like($@, qr/^100/, "exceptionspec 3");

my $ex = new default_args::Except($false);

my $hit = 0;
eval { $ex->exspec(); $hit = 1; };
# a zero was thrown, an exception occurred, but $@ is false
is($hit, 0, "exspec 1");
eval { $ex->exspec(-1) };
like($@, qr/^ciao/, "exspec 2");
eval { $ex->exspec(100) };
like($@, qr/^100/, "exspec 3");
eval { $ex = default_args::Except->new($true) };
like($@, qr/-1/, "Except constructor 1");
eval { $ex = default_args::Except->new($true, -2) };
like($@, qr/-2/, "Except constructor 2");

#Default parameters in static class methods
is(default_args::Statics::staticmethod(), 60, "staticmethod 1");
is(default_args::Statics::staticmethod(100), 150, "staticmethod 2");
is(default_args::Statics::staticmethod(100,200,300), 600, "staticmethod 3");

my $tricky = new default_args::Tricky();
is($tricky->privatedefault(), 200, "privatedefault");
is($tricky->protectedint(), 2000, "protectedint");
is($tricky->protecteddouble(), 987.654, "protecteddouble");
is($tricky->functiondefault(), 500, "functiondefault");
is($tricky->contrived(), 'X', "contrived");
is(default_args::constructorcall()->{val}, -1, "constructorcall test 1");
is(default_args::constructorcall(new default_args::Klass(2222))->{val},
   2222, "constructorcall test 2");
is(default_args::constructorcall(new default_args::Klass())->{val},
   -1, "constructorcall test 3");

# const methods 
my $cm = new default_args::ConstMethods();
is($cm->coo(), 20, "coo test 1");
is($cm->coo(1.0), 20, "coo test 2");
