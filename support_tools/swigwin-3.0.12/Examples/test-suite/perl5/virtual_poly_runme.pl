#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 8;
BEGIN { use_ok('virtual_poly') }
require_ok('virtual_poly');

my $d = virtual_poly::NDouble->new(3.5);
my $i = virtual_poly::NInt->new(2);

#
# the copy methods return the right polymorphic types
# 
my $dc = $d->copy();
my $ic = $i->copy();

is($d->get(), $dc->get());

is($i->get(), $ic->get());

virtual_poly::incr($ic);

is($i->get() + 1, $ic->get());

my $dr = $d->ref_this();
is($d->get(), $dr->get());


#
# 'narrowing' also works
#
my $ddc = virtual_poly::NDouble::narrow($d->nnumber());
is($d->get, $ddc->get());

my $dic = virtual_poly::NInt::narrow($i->nnumber());
is($i->get(), $dic->get());
