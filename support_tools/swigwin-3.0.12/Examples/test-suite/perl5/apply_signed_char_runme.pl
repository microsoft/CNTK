#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 12;
BEGIN { use_ok('apply_signed_char') }
require_ok('apply_signed_char');

# adapted from ../java/apply_signed_char_runme.java

my $smallnum = -127;
is(apply_signed_char::CharValFunction($smallnum), $smallnum);
is(apply_signed_char::CCharValFunction($smallnum), $smallnum);
is(apply_signed_char::CCharRefFunction($smallnum), $smallnum);

$apply_signed_char::globalchar = $smallnum;
is($apply_signed_char::globalchar, $smallnum);
is($apply_signed_char::globalconstchar, -110);

my $d = new apply_signed_char::DirectorTest();
is($d->CharValFunction($smallnum), $smallnum);
is($d->CCharValFunction($smallnum), $smallnum);
is($d->CCharRefFunction($smallnum), $smallnum);

$d->{memberchar} = $smallnum;
is($d->{memberchar}, $smallnum);
is($d->{memberconstchar}, -112);
