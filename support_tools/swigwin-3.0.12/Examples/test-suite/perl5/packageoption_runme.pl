#!/usr/bin/perl -w
use strict;
use Test::More tests => 4;

BEGIN { use_ok('packageoption_a'); }
BEGIN { use_ok('packageoption_b'); }

# Workaround for 
#   ok( not (expression) , "test description" );
# does not working in older versions of Perl, eg 5.004_04
sub ok_not ($;$) {
    my($test, $name) = @_;
    $test = not $test;
    ok($test, $name);
}

my $a = CommonPackage::A->new();

isa_ok($a, 'CommonPackage::A');

my $b = CommonPackage::B->new();

isa_ok($b, 'CommonPackage::B');
       
