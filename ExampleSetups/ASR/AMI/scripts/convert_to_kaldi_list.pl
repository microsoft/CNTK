#!/usr/bin/env perl 
#==============================================================================
#
#         FILE: convert_to_kaldi_list.pl
#
#        USAGE: ./convert_to_kaldi_list.pl  
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
#      CREATED: 09/03/2013 01:55:47 PM
#     REVISION: ---
#===============================================================================

use strict;
use warnings;
use utf8;

my $uttList = $ARGV[1];
my $uttMap = $ARGV[0];
my $kaldi_list = $ARGV[2];
my $out_dir = $ARGV[3];

my %uttMapHash = ();
my $useMap = 1;
if ($uttMap=~/noMap/)
{
    $useMap = 0;
}else
{
    open UTTMAP, $uttMap or die "Cannot open $uttMap\n";
    while (my $line = <UTTMAP>)
    {
        my ($key, $value) = split (/\s+/, $line);
        $uttMapHash{$key} = $value;
    }
}
close(UTTMAP);

open UTTLIST, $uttList or die "Cannot open $uttList\n";
my %uttListHash = ();
while (my $line = <UTTLIST>)
{
    chomp($line);
    my @temp = split (/\//, $line);
    my ($key, $tmp) = split (/\./, $temp[@temp-1]);
    $uttListHash{$key} = $line;
}
close(UTTLIST);

open KLIST, $kaldi_list or die "Cannot open $kaldi_list\n";
open OUTPUT_IN, ">$out_dir/sls.scp";
while (my $line = <KLIST>)
{
    my @temp = split (/\s+/, $line);
    if ($useMap==1)
    {
        if (exists $uttMapHash{$temp[0]})
        {
            my $new = $uttListHash{$uttMapHash{$temp[0]}};
            my $ori = $new;
            $new =~ s/$uttMapHash{$temp[0]}/$temp[0]/g;
            #system("cp $uttListHash{$uttMapHash{$temp[0]}} $new"); 
            print "$uttListHash{$uttMapHash{$temp[0]}} $new\n";
            print OUTPUT_IN "$temp[0] $ori\n";
        }
    }else
    {
        if (exists $uttListHash{$temp[0]})
        {
            print OUTPUT_IN "$temp[0] $uttListHash{$temp[0]}\n";
        }
    }
}
close(KLIST);
