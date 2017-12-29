use strict;
use warnings;
use Test::More tests => 4;
BEGIN { use_ok('apply_strings') }
require_ok('apply_strings');

my $TEST_MESSAGE = "A message from target language to the C++ world and back again.";

is(apply_strings::UCharFunction($TEST_MESSAGE), $TEST_MESSAGE, "UCharFunction"); 

is(apply_strings::SCharFunction($TEST_MESSAGE), $TEST_MESSAGE, "SCharFunction");
