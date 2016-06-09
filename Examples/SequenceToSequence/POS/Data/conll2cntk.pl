#! /usr/bin/env perl

use warnings;
use strict;

#Confidence NN B-NP
#in IN B-PP
#the DT B-NP
#pound NN I-NP
#is VBZ B-VP
#widely RB I-VP
#expected VBN I-VP
#to TO I-VP
#take VB I-VP
#another DT B-NP
#sharp JJ I-NP
#dive NN I-NP
#if IN B-SBAR
#trade NN B-NP
#figures NNS I-NP
#for IN B-PP
#September NNP B-NP
#, , O
#due JJ B-ADJP
#for IN B-PP
#release NN B-NP
#tomorrow NN B-NP
#, , O
#fail VB B-VP
#to TO I-VP
#show VB I-VP
#a DT B-NP
#substantial JJ I-NP
#improvement NN I-NP
#from IN B-PP
#July NNP B-NP
#and CC I-NP
#August NNP I-NP
#'s POS B-NP
#near-record JJ I-NP
#deficits NNS I-NP
#. . O
#
#Chancellor NNP O
#of IN B-PP

my $vocabfile = shift @ARGV or die;
my $labelfile = shift @ARGV or die;

my %vocab;
my %label;

sub encode ($$)
  {
    my ($v, $dict) = @_;

    $dict->{$v} ||= scalar keys %$dict;
    return $dict->{$v};
  }

my $sid = 0;
while (defined ($_ = <>))
  {
    chomp;
    do { ++$sid; next; } unless /\S/;

    my ($token, $pos, undef) = split /\s+/, $_;
    my $enctok = encode ($token, \%vocab);
    my $encpos = encode ($pos, \%label);

    print "$sid |Word ${enctok}:1 |Part ${encpos}:1\n";
  }

my $vocabfh = new IO::File $vocabfile, "w" or die "$vocabfile: $!";

while (my ($k, $v) = each %vocab)
  {
    print $vocabfh "$k $v\n";
  }

my $labelfh = new IO::File $labelfile, "w" or die "$labelfile: $!";

while (my ($k, $v) = each %label)
  {
    print $labelfh "$k $v\n";
  }
