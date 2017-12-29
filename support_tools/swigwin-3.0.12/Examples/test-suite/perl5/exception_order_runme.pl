#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 7;
BEGIN { use_ok('exception_order') }
require_ok('exception_order');

# adapted from ../python/exception_order_runme.py

my $a = exception_order::A->new();

eval { $a->foo() };
isa_ok($@, "exception_order::E1");

eval { $a->bar() };
isa_ok($@, "exception_order::E2");

eval { $a->foobar() };
like($@, qr/\bpostcatch unknown\b/);

eval { $a->barfoo(1) };
isa_ok($@, "exception_order::E1");

eval { $a->barfoo(2) };
isa_ok($@, "exception_order::E2");
