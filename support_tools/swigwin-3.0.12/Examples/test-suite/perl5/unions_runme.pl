use strict;
use warnings;
use Test::More tests => 7;
# This is the union runtime testcase. It ensures that values within a 
# union embedded within a struct can be set and read correctly.

BEGIN { use_ok('unions') }
require_ok('unions');

# Create new instances of SmallStruct and BigStruct for later use
my $small = new unions::SmallStruct();
$small->{jill} = 200;

my $big = new unions::BigStruct();
$big->{smallstruct} = $small;
$big->{jack} = 300;

# Use SmallStruct then BigStruct to setup EmbeddedUnionTest.
# Ensure values in EmbeddedUnionTest are set correctly for each.
my $eut = new unions::EmbeddedUnionTest();

# First check the SmallStruct in EmbeddedUnionTest
$eut->{number} = 1;
$eut->{uni}->{small} = $small;
my $Jill1 = $eut->{uni}->{small}->{jill};
is($Jill1, 200, "eut.uni.small.jill");

my $Num1 = $eut->{number};
is($Num1, 1, "test2 eut.number");

# Secondly check the BigStruct in EmbeddedUnionTest
$eut->{number} = 2;
$eut->{uni}->{big} = $big;
my $Jack1 = $eut->{uni}->{big}->{jack};
is($Jack1, 300, "test3 eut.uni.big.jack");

my $Jill2 = $eut->{uni}->{big}->{smallstruct}->{jill};
is($Jill2, 200, "test4 eut.uni.big.smallstruct.jill");

my $Num2 = $eut->{number};
is($Num2,  2, "test5 eut.number");

