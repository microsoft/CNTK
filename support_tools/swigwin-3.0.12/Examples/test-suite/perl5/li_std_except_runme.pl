#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 22;
BEGIN { use_ok('li_std_except') }
require_ok('li_std_except');

# adapted from ../java/li_std_except_runme.java

# these are not prescriptive tests, they just match the error classes I
# found are currently being issued, we may want to provide a more
# granular error api later, so don't let these tests stop code
# improvements.

my $test = new li_std_except::Test();
eval { $test->throw_bad_exception() };
like($@, qr/\bSystemError\b/, "throw_bad_exception");
eval { $test->throw_domain_error() };
like($@, qr/\bValueError\b/, "throw_domain_error");
like($@, qr/\boops\b/, "throw_domain_error message");
eval { $test->throw_exception() };
like($@, qr/\bSystemError\b/, "throw_exception");
eval { $test->throw_invalid_argument() };
like($@, qr/\bValueError\b/, "throw_invalid_argument");
like($@, qr/\boops\b/, "throw_invalid_argument message");
eval { $test->throw_length_error() };
like($@, qr/\bIndexError\b/, "throw_length_error");
like($@, qr/\boops\b/, "throw_length_error message");
eval { $test->throw_logic_error() };
like($@, qr/\bRuntimeError\b/, "throw_logic_error");
like($@, qr/\boops\b/, "throw_logic_error message");
eval { $test->throw_out_of_range() };
like($@, qr/\bIndexError\b/, "throw_out_of_range");
like($@, qr/\boops\b/, "throw_out_of_range message");
eval { $test->throw_overflow_error() };
like($@, qr/\bOverflowError\b/, "throw_overflow_error");
like($@, qr/\boops\b/, "throw_overflow_error message");
eval { $test->throw_range_error() };
like($@, qr/\bOverflowError\b/, "throw_range_error");
like($@, qr/\boops\b/, "throw_range_error message");
eval { $test->throw_runtime_error() };
like($@, qr/\bRuntimeError\b/, "throw_runtime_error");
like($@, qr/\boops\b/, "throw_runtime_error message");
eval { $test->throw_underflow_error() };
like($@, qr/\bOverflowError\b/, "throw_underflow_error");
like($@, qr/\boops\b/, "throw_underflow_error message");
