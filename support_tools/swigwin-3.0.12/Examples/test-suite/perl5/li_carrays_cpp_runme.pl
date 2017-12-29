#!/usr/bin/perl
use strict;
use warnings;
use Test::More tests => 34;
BEGIN { use_ok('li_carrays_cpp') }
require_ok('li_carrays_cpp');

# array_class
{
  my $length = 5;
  my $xyArray = new li_carrays_cpp::XYArray($length);
  for (my $i=0; $i<$length; $i++) {
    my $xy = $xyArray->getitem($i);
    $xy->{x} = $i*10;
    $xy->{y} = $i*100;
    $xyArray->setitem($i, $xy);
  }
  for (my $i=0; $i<$length; $i++) {
    is($xyArray->getitem($i)->{x}, $i*10);
    is($xyArray->getitem($i)->{y}, $i*100);
  }
}

{
  # global array variable
  my $length = 3;
  my $xyArrayPointer = $li_carrays_cpp::globalXYArray;
  my $xyArray = li_carrays_cpp::XYArray::frompointer($xyArrayPointer);
  for (my $i=0; $i<$length; $i++) {
    my $xy = $xyArray->getitem($i);
    $xy->{x} = $i*10;
    $xy->{y} = $i*100;
    $xyArray->setitem($i, $xy);
  }
  for (my $i=0; $i<$length; $i++) {
    is($xyArray->getitem($i)->{x}, $i*10);
    is($xyArray->getitem($i)->{y}, $i*100);
  }
}

# array_functions
{
  my $length = 5;
  my $abArray = li_carrays_cpp::new_ABArray($length);
  for (my $i=0; $i<$length; $i++) {
    my $ab = li_carrays_cpp::ABArray_getitem($abArray, $i);
    $ab->{a} = $i*10;
    $ab->{b} = $i*100;
	li_carrays_cpp::ABArray_setitem($abArray, $i, $ab);
  }
  for (my $i=0; $i<$length; $i++) {
    is(li_carrays_cpp::ABArray_getitem($abArray, $i)->{a}, $i*10);
    is(li_carrays_cpp::ABArray_getitem($abArray, $i)->{b}, $i*100);
  }
  li_carrays_cpp::delete_ABArray($abArray);
}

{
  # global array variable
  my $length = 3;
  my $abArray = $li_carrays_cpp::globalABArray;
  for (my $i=0; $i<$length; $i++) {
    my $ab = li_carrays_cpp::ABArray_getitem($abArray, $i);
    $ab->{a} = $i*10;
    $ab->{b} = $i*100;
	li_carrays_cpp::ABArray_setitem($abArray, $i, $ab);
  }
  for (my $i=0; $i<$length; $i++) {
    is(li_carrays_cpp::ABArray_getitem($abArray, $i)->{a}, $i*10);
    is(li_carrays_cpp::ABArray_getitem($abArray, $i)->{b}, $i*100);
  }
}
