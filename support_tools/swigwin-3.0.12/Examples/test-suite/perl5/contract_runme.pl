#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 32;
BEGIN { use_ok('contract') }
require_ok('contract');

# adapted from ../python/contract_runme.py
{
	ok(contract::test_preassert(1,2), "good preassertion");
	eval { contract::test_preassert(-1) };
	like($@, qr/\bRuntimeError\b/, "bad preassertion");

	ok(contract::test_postassert(3), "good postassertion");
	eval { contract::test_postassert(-3) };
	like($@, qr/\bRuntimeError\b/, "bad postassertion");

	ok(contract::test_prepost(2,3), "good prepost");
	ok(contract::test_prepost(5,-4), "good prepost");
	eval { contract::test_prepost(-3,4); };
	like($@, qr/\bRuntimeError\b/, "bad preassertion");
	eval { contract::test_prepost(4,-10) };
	like($@, qr/\bRuntimeError\b/, "bad postassertion");
}
{
	my $f = contract::Foo->new();
	ok($f->test_preassert(4,5), "method pre");
	eval { $f->test_preassert(-2,3) };
	like($@, qr/\bRuntimeError\b/, "method pre bad");

	ok($f->test_postassert(4), "method post");
	eval { $f->test_postassert(-4) };
	like($@, qr/\bRuntimeError\b/, "method post bad");

	ok($f->test_prepost(3,4), "method prepost");
	ok($f->test_prepost(4,-3), "method prepost");
	eval { $f->test_prepost(-4,2) };
	like($@, qr/\bRuntimeError\b/, "method pre bad");
	eval { $f->test_prepost(4,-10) };
	like($@, qr/\bRuntimeError\b/, "method post bad");
}
{
	ok(contract::Foo::stest_prepost(4,0), "static method prepost");
	eval { contract::Foo::stest_prepost(-4,2) };
	like($@, qr/\bRuntimeError\b/, "static method pre bad");
	eval { contract::Foo::stest_prepost(4,-10) };
	like($@, qr/\bRuntimeError\b/, "static method post bad");
}
{
	my $b = contract::Bar->new();
	eval { $b->test_prepost(2,-4) };
	like($@, qr/\bRuntimeError\b/, "inherit pre bad");
}
{
	my $d = contract::D->new();
	eval { $d->foo(-1,1,1,1,1) };
	like($@, qr/\bRuntimeError\b/, "inherit pre D");
	eval { $d->foo(1,-1,1,1,1) };
	like($@, qr/\bRuntimeError\b/, "inherit pre D");
	eval { $d->foo(1,1,-1,1,1) };
	like($@, qr/\bRuntimeError\b/, "inherit pre D");
	eval { $d->foo(1,1,1,-1,1) };
	like($@, qr/\bRuntimeError\b/, "inherit pre D");
	eval { $d->foo(1,1,1,1,-1) };
	like($@, qr/\bRuntimeError\b/, "inherit pre D");

	eval { $d->bar(-1,1,1,1,1) };
	like($@, qr/\bRuntimeError\b/, "inherit pre D");
	eval { $d->bar(1,-1,1,1,1) };
	like($@, qr/\bRuntimeError\b/, "inherit pre D");
	eval { $d->bar(1,1,-1,1,1) };
	like($@, qr/\bRuntimeError\b/, "inherit pre D");
	eval { $d->bar(1,1,1,-1,1) };
	like($@, qr/\bRuntimeError\b/, "inherit pre D");
	eval { $d->bar(1,1,1,1,-1) };
	like($@, qr/\bRuntimeError\b/, "inherit pre D");
}

