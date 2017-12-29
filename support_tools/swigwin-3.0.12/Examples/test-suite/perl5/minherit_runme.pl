#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 38;
BEGIN { use_ok('minherit') }
require_ok('minherit');

# adapted from ../python/minherit_runme.py

my $a = minherit::Foo->new();
my $b = minherit::Bar->new();
my $c = minherit::FooBar->new();
my $d = minherit::Spam->new();

is($a->xget(), 1);

is($b->yget(), 2);

is($c->xget(), 1);
is($c->yget(), 2);
is($c->zget(), 3);

is($d->xget(), 1);
is($d->yget(), 2);
is($d->zget(), 3);
is($d->wget(), 4);

is(minherit::xget($a), 1);

is(minherit::yget($b), 2);

is(minherit::xget($c), 1);
is(minherit::yget($c), 2);
is(minherit::zget($c), 3);

is(minherit::xget($d), 1);
is(minherit::yget($d), 2);
is(minherit::zget($d), 3);
is(minherit::wget($d), 4);

# Cleanse all of the pointers and see what happens

my $aa = minherit::toFooPtr($a);
my $bb = minherit::toBarPtr($b);
my $cc = minherit::toFooBarPtr($c);
my $dd = minherit::toSpamPtr($d);

is($aa->xget, 1);

is($bb->yget(), 2);

is($cc->xget(), 1);
is($cc->yget(), 2);
is($cc->zget(), 3);

is($dd->xget(), 1);
is($dd->yget(), 2);
is($dd->zget(), 3);
is($dd->wget(), 4);

is(minherit::xget($aa), 1);

is(minherit::yget($bb), 2);

is(minherit::xget($cc), 1);
is(minherit::yget($cc), 2);
is(minherit::zget($cc), 3);

is(minherit::xget($dd), 1);
is(minherit::yget($dd), 2);
is(minherit::zget($dd), 3);
is(minherit::wget($dd), 4);
