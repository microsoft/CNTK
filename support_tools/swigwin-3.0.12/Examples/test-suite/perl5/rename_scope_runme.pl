#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 5;
BEGIN { use_ok('rename_scope') }
require_ok('rename_scope');

# adapted from ../python/rename_scope_runme.py

my $a = rename_scope::Natural_UP->new();
is($a->rtest(), 1);
my $b = rename_scope::Natural_BP->new();
is($b->rtest(), 1);

isa_ok(rename_scope->can('equals'), 'CODE');
