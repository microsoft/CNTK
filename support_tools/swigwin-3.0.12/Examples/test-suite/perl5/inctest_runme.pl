#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 3;
BEGIN { use_ok('inctest') }
require_ok('inctest');

my $things = inctest::MY_THINGS->new();
my $i = 0;
$things->{IntegerMember} = $i;
my $d = $things->{DoubleMember};
ok(1);
