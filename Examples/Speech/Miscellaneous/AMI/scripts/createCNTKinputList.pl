#!/usr/bin/env perl 
#===============================================================================
#
#         FILE: createCNTKinput.pl
#
#        USAGE: ./createCNTKinput.pl  
#
#  DESCRIPTION: 
#
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
#       AUTHOR: YOUR NAME (), 
# ORGANIZATION: 
#      VERSION: 1.0
#      CREATED: 02/16/2015 04:18:01 PM
#     REVISION: ---
#===============================================================================

use warnings;
use utf8;

my $inputDir = $ARGV[0];
my $inputSLS = $ARGV[1];
my $hash_map = ();
open FILE, $inputSLS or die;
while (my $line = <FILE>)
{
    my ($key, $value) = split (/\s+/, $line);
    $hash_map{$key} = $value;
}
close (FILE);
open FILE, "$inputDir/feats.scp" or die;
open OUT2, ">$inputDir/cntkOutput.scp" or die;
open OUT, ">$inputDir/cntkInput.scp" or die;
while (my $line = <FILE>)
{
    my ($key, $value) = split (/\s+/, $line);
    ($outline,$pos) = split(/\.output/,$hash_map{$key});
    print OUT2 "$outline.output\n";
    print OUT "$outline\n";
}

close(FILE);
close(OUT);
close(OUT2);
