use strict;
use warnings;
use Test::More tests => 5;
BEGIN { use_ok('char_strings') }
require_ok('char_strings');

my $val1 = "100";
is(char_strings::CharPingPong($val1), "100", 'cstr1');

my $val2 = "greetings";
is(char_strings::CharPingPong($val2), "greetings", 'cstr2');

# SF#2564192
"this is a test" =~ /(\w+)$/;
is(char_strings::CharPingPong($1), "test", "handles Magical");
