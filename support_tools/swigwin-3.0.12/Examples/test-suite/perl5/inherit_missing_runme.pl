#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 5;
BEGIN { use_ok('inherit_missing') }
require_ok('inherit_missing');

# adapted from ../python/inherit_missing_runme.py

my $a = inherit_missing::new_Foo();
my $b = inherit_missing::Bar->new();
my $c = inherit_missing::Spam->new();

is(inherit_missing::do_blah($a), "Foo::blah");

is(inherit_missing::do_blah($b), "Bar::blah");

is(inherit_missing::do_blah($c), "Spam::blah");

inherit_missing::delete_Foo($a);
