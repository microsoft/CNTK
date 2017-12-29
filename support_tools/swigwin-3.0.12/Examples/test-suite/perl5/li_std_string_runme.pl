use strict;
use warnings;
use Test::More tests => 30;
BEGIN { use_ok('li_std_string') }
require_ok('li_std_string');

use Devel::Peek;
# Checking expected use of %typemap(in) std::string {}
li_std_string::test_value("Fee");

# Checking expected result of %typemap(out) std::string {}
is(li_std_string::test_value("Fi"), "Fi", "Test 1");


# Verify type-checking for %typemap(in) std::string {}
eval { li_std_string::test_value(undef) };
like($@, qr/\bTypeError\b/, "Test 2");

# Checking expected use of %typemap(in) const std::string & {}
li_std_string::test_const_reference("Fo");

# Checking expected result of %typemap(out) const std::string& {}
is(li_std_string::test_const_reference("Fum"), "Fum", "Test 3");

# Verify type-checking for %typemap(in) const std::string & {}
eval { li_std_string::test_const_reference(undef) };
like($@, qr/\bValueError\b/, "Test 4");

#
# Input and output typemaps for pointers and non-const references to
# std::string are *not* supported; the following tests confirm
# that none of these cases are slipping through.
#

my $stringPtr = undef;

$stringPtr = li_std_string::test_pointer_out();

li_std_string::test_pointer($stringPtr);

$stringPtr = li_std_string::test_const_pointer_out();

li_std_string::test_const_pointer($stringPtr);

$stringPtr = li_std_string::test_reference_out();

li_std_string::test_reference($stringPtr);

# Check throw exception specification
eval { li_std_string::test_throw() };
like($@, qr/^test_throw message/, "Test 5");
{ local $TODO = "why is the error not a Perl string?";
eval { li_std_string::test_const_reference_throw() };
is($@, "<some kind of string>", "Test 6");
}

# Global variables
my $s = "initial string";
is($li_std_string::GlobalString2, "global string 2", "GlobalString2 test 1");
$li_std_string::GlobalString2 = $s;
is($li_std_string::GlobalString2, $s, "GlobalString2 test 2");
is($li_std_string::ConstGlobalString, "const global string", "ConstGlobalString test");

# Member variables
my $myStructure = new li_std_string::Structure();
is($myStructure->{MemberString2}, "member string 2", "MemberString2 test 1");
$myStructure->{MemberString2} = $s;
is($myStructure->{MemberString2}, $s, "MemberString2 test 2");
is($myStructure->{ConstMemberString}, "const member string", "ConstMemberString test");

is($li_std_string::Structure::StaticMemberString2, "static member string 2", "StaticMemberString2 test 1");
$li_std_string::Structure::StaticMemberString2 = $s;
is($li_std_string::Structure::StaticMemberString2, $s, "StaticMemberString2 test 2");
is($li_std_string::Structure::ConstStaticMemberString, "const static member string", "ConstStaticMemberString test");

is(li_std_string::test_reference_input("hello"), "hello", "reference_input");

is(li_std_string::test_reference_inout("hello"), "hellohello", "reference_inout");


no strict;
my $gen1 = new li_std_string::Foo();
is($gen1->test(1), 2, "ulonglong");
is($gen1->test("1"), "11", "ulonglong");
is($gen1->testl(12345), 12346, "ulonglong small number");
# Note: 32 bit builds of perl will fail this test as the number is stored internally in scientific notation 
# (USE_64_BIT_ALL probably needs defining when building Perl in order to avoid this)
SKIP: {
	skip "this Perl does not seem to do 64 bit ints", 1
		if 9234567890121111114 - 9234567890121111113 != 1;
	local $TODO;
	use Config;
	$TODO = "if we're lucky this might work" unless $Config{use64bitall};
	is(eval { $gen1->testl(9234567890121111113) }, 9234567890121111114, "ulonglong big number");
	# TODO: I suspect we can get by with "use64bitint", but I'll have to
	# work that out later. -talby
}
is($gen1->testl("9234567890121111113"), "9234567890121111114", "ulonglong big number");


is(li_std_string::stdstring_empty(), "", "stdstring_empty");

is(li_std_string::c_empty(),  "", "c_empty");


is(li_std_string::c_null(), undef, "c_empty");


is(li_std_string::get_null(li_std_string::c_null()), undef, "c_empty");

is(li_std_string::get_null(li_std_string::c_empty()), "non-null", "c_empty");

is(li_std_string::get_null(li_std_string::stdstring_empty()), "non-null", "stdstring_empty");
