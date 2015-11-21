#!/usr/bin/env perl 
#===============================================================================
#
#         FILE: convert_kaldi_label.pl
#
#        USAGE: ./convert_kaldi_label.pl  
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
#      CREATED: 09/18/2013 02:38:26 PM
#     REVISION: ---
#===============================================================================

use strict;
use warnings;
use utf8;

my $aliDir = $ARGV[0];
my $featureLoc = $ARGV[1];
my $newDir = $ARGV[2];
#my $phoneMap = $ARGV[3];
my $posix = $ARGV[3];

my $CILabels=0;


print "$posix\n";
opendir ALID, $aliDir or die;
mkdir("$newDir");
open OUTLAB, ">$newDir/labels.merged" or die;
my %hashMap=();
while (my $file = readdir ALID)
{
    if ($file =~ /train_ali/ && !($file =~ /sw/))
    {
        open FILE, "$aliDir/$file" or die;
        print "$file\n";

        $file =~ s/train_ali/phone.tra/g;
        open CILABEL, "$aliDir/$file";

        while (my $line = <FILE>)
        {
            my ($key, @labels) = split (/\s+/, $line);
            my @temp = split (/-|_/, $key);
			
            if (-e "$featureLoc/$key.fea")
            {
                for (my $i = 0; $i < @labels; $i++)
                {
                    $labels[$i] = $labels[$i]+1;
                }         
                open LABELFILE, ">$featureLoc/$key.label$posix" or die;
                print LABELFILE join("\n",@labels);
                #close(LABELFILE);
            }
        }
        close(CILABEL);
        close(FILE);
    }
}
foreach my $key (keys %hashMap)
{
    print "$key $hashMap{$key}\n";
}
close (OUTLAB);
