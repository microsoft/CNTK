#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 6;
BEGIN { use_ok('sizet') }
require_ok('sizet');

# adapted from ../java/sizet_runme.java

my $s = 2000;
$s = sizet::test1($s + 1);
is($s, 2001, 'test1');
$s = sizet::test1($s + 1);
is($s, 2002, 'test2');
$s = sizet::test1($s + 1);
is($s, 2003, 'test3');
$s = sizet::test1($s + 1);
is($s, 2004, 'test4');
