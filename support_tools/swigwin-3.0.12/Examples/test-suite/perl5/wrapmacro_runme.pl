#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 27;
BEGIN { use_ok('wrapmacro') }
require_ok('wrapmacro');

# adapted from ../python/wrapmacro_runme.py

my $a = 2;
my $b = -1;
is(wrapmacro::maximum($a,$b), 2);
is(wrapmacro::maximum($a/7.0, -$b*256), 256);
is(wrapmacro::GUINT16_SWAP_LE_BE_CONSTANT(1), 256);

# some of this section is duplication of above tests, but I want to see
# parity with the coverage in overload_simple_runme.pl.

sub check {
  my($args, $rslt) = @_;
  my $s = defined $rslt ? $rslt : '*boom*';
  is(eval("wrapmacro::maximum($args)"), $rslt, "max($args) => $s");
}

# normal use patterns
check("0, 11", 11);
check("0, 11.0", 11);
check("0, '11'", 11);
check("0, '11.0'", 11);
check("11, -13", 11);
check("11, -13.0", 11);
{ local $TODO = 'strtoull() handles /^\s*-\d+$/ amusingly';
check("11, '-13'", 11);
}
check("11, '-13.0'", 11);

# TypeError explosions
check("0, ' '", undef);
check("0, ' 11 '", undef);
check("0, \\*STDIN", undef);
check("0, []", undef);
check("0, {}", undef);
check("0, sub {}", undef);

# regression cases
{ local $TODO = 'strtol() and friends have edge cases we should guard against';
check("-11, ''", undef);
check("0, ' 11'", undef);
check("0, ' 11.0'", undef);
check("-13, ' -11.0'", undef);
check("0, \"11\x{0}\"", undef);
check("0, \"\x{0}\"", undef);
check("0, \"\x{9}11\x{0}this is not eleven.\"", undef);
check("0, \"\x{9}11.0\x{0}this is also not eleven.\"", undef);
}
