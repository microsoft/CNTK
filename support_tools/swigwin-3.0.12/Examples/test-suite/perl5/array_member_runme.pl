#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 18;
BEGIN { use_ok('array_member') }
require_ok('array_member');

my $f = array_member::Foo->new();
$f->{data} = $array_member::global_data;

for(my $i=0; $i<8; $i++) {
	is( array_member::get_value($f->{data},$i),
		array_member::get_value($array_member::global_data,$i),
		"array assignment");
}

for(my $i=0; $i<8; $i++) {
	array_member::set_value($f->{data},$i,-$i);
}

$array_member::global_data = $f->{data};

for(my $i=0; $i<8; $i++) {
	is(array_member::get_value($f->{data},$i),
		array_member::get_value($array_member::global_data,$i),
		"array assignment");
}

