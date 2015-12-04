#!/usr/bin/env perl 
#===============================================================================
#
#         FILE: rand_select_data.pl
#
#        USAGE: ./rand_select_data.pl  
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
#      CREATED: 09/12/2013 05:21:58 PM
#     REVISION: ---
#===============================================================================

use strict;
use warnings;

use List::Util 'shuffle';
use utf8;

my $feat_list = $ARGV[0];
my $label_list = $ARGV[1];
my $percent = $ARGV[2];



if (@ARGV == 2)
{
    $percent = 10;
}

open FEAT, "$feat_list" or die "Cannot open $feat_list\n";
open LABEL, "$label_list" or die "Cannot open $label_list\n";

open OutTrain, ">$feat_list.train";
open OutTest, ">$feat_list.test";
open OutTrainLabel, ">$label_list.train.label";
open OutTestLabel, ">$label_list.test.label";

my %train = ();

while (my $featFile = <FEAT>)
{
    my $labelFile = <LABEL>;
    my $random_number = int(rand(10));

    if ($random_number == 0)
    {
        print OutTest $featFile;
        print OutTestLabel $labelFile;
    }else
    {
        $train{$featFile} = $labelFile;
    }

}

foreach my $feat (shuffle keys %train)
{
    print OutTrain $feat;
    print OutTrainLabel $train{$feat}

}

close FEAT;
close LABEL;
close OutTrain;
close OutTest;
close OutTrainLabel;
close OutTestLabel;
