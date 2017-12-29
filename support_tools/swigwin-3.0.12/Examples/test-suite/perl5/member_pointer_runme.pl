use strict;
use warnings;
use Test::More tests => 9;
# member_pointer using pointers to member functions

BEGIN { use_ok('member_pointer') }
require_ok('member_pointer');

sub check($;$;$) {
  my($what, $expected, $actual) = @_;
  if ($expected != $actual) {
    die ("Failed: $what Expected: $expected Actual: $actual");
  }
}

# Get the pointers

my $area_pt = member_pointer::areapt();
my $perim_pt = member_pointer::perimeterpt();

# Create some objects

my $s = new member_pointer::Square(10);

# Do some calculations

is(100.0, member_pointer::do_op($s,$area_pt), "Square area");
is(40.0, member_pointer::do_op($s,$perim_pt), "Square perim");
no strict;

my $memberPtr = $member_pointer::areavar;
$memberPtr = $member_pointer::perimetervar;

# Try the variables
is(100.0, member_pointer::do_op($s,$member_pointer::areavar), "Square area");
is(40.0, member_pointer::do_op($s,$member_pointer::perimetervar), "Square perim");

# Modify one of the variables
$member_pointer::areavar = $perim_pt;

is(40.0, member_pointer::do_op($s,$member_pointer::areavar), "Square perimeter");

# Try the constants

$memberPtr = $member_pointer::AREAPT;
$memberPtr = $member_pointer::PERIMPT;
$memberPtr = $member_pointer::NULLPT;

is(100.0, member_pointer::do_op($s,$member_pointer::AREAPT), "Square area");
is(40.0, member_pointer::do_op($s,$member_pointer::PERIMPT), "Square perim");

