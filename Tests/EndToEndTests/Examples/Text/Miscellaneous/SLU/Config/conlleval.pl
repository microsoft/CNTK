#!/usr/bin/perl -w
# conlleval: evaluate result of processing CoNLL-2000 shared task
# usage:     conlleval [-l] [-r] [-d delimiterTag] [-o oTag] < file
#            README: http://cnts.uia.ac.be/conll2000/chunking/output.html
# options:   l: generate LaTeX output for tables like in
#               http://cnts.uia.ac.be/conll2003/ner/example.tex
#            r: accept raw result tags (without B- and I- prefix;
#                                       assumes one word per chunk)
#            d: alternative delimiter tag (default is single space)
#            o: alternative outside tag (default is O)
# note:      the file should contain lines with items separated
#            by $delimiter characters (default space). The final
#            two items should contain the correct tag and the 
#            guessed tag in that order. Sentences should be
#            separated from each other by empty lines or lines
#            with $boundary fields (default -X-).
# url:       http://lcg-www.uia.ac.be/conll2000/chunking/
# started:   1998-09-25
# version:   2004-01-26
# author:    Erik Tjong Kim Sang <erikt@uia.ua.ac.be>

use strict;

my $false = 0;
my $true = 42;

my $boundary = "-X-";     # sentence boundary
my $correct;              # current corpus chunk tag (I,O,B)
my $correctChunk = 0;     # number of correctly identified chunks
my $correctTags = 0;      # number of correct chunk tags
my $correctType;          # type of current corpus chunk tag (NP,VP,etc.)
my $delimiter = " ";      # field delimiter
my $FB1 = 0.0;            # FB1 score (Van Rijsbergen 1979)
my $firstItem;            # first feature (for sentence boundary checks)
my $foundCorrect = 0;     # number of chunks in corpus
my $foundGuessed = 0;     # number of identified chunks
my $guessed;              # current guessed chunk tag
my $guessedType;          # type of current guessed chunk tag
my $i;                    # miscellaneous counter
my $inCorrect = $false;   # currently processed chunk is correct until now
my $lastCorrect = "O";    # previous chunk tag in corpus
my $latex = 0;            # generate LaTeX formatted output
my $lastCorrectType = ""; # type of previously identified chunk tag
my $lastGuessed = "O";    # previously identified chunk tag
my $lastGuessedType = ""; # type of previous chunk tag in corpus
my $lastType;             # temporary storage for detecting duplicates
my $line;                 # line
my $nbrOfFeatures = -1;   # number of features per line
my $precision = 0.0;      # precision score
my $oTag = "O";           # outside tag, default O
my $raw = 0;              # raw input: add B to every token
my $recall = 0.0;         # recall score
my $tokenCounter = 0;     # token counter (ignores sentence breaks)

my %correctChunk = ();    # number of correctly identified chunks per type
my %foundCorrect = ();    # number of chunks in corpus per type
my %foundGuessed = ();    # number of identified chunks per type

my @features;             # features on line
my @sortedTypes;          # sorted list of chunk type names

# sanity check
while (@ARGV and $ARGV[0] =~ /^-/) {
   if ($ARGV[0] eq "-l") { $latex = 1; shift(@ARGV); }
   elsif ($ARGV[0] eq "-r") { $raw = 1; shift(@ARGV); }
   elsif ($ARGV[0] eq "-d") { 
      shift(@ARGV); 
      if (not defined $ARGV[0]) { 
         die "conlleval: -d requires delimiter character"; 
      }
      $delimiter = shift(@ARGV);
   } elsif ($ARGV[0] eq "-o") {
      shift(@ARGV);
      if (not defined $ARGV[0]) {
         die "conlleval: -o requires delimiter character";
      }
      $oTag = shift(@ARGV);
   } else { die "conlleval: unknown argument $ARGV[0]\n"; }
}
if (@ARGV) { die "conlleval: unexpected command line argument\n"; }
# process input
while (<STDIN>) {
   chomp($line = $_);
   @features = split(/$delimiter/,$line);
   if ($nbrOfFeatures < 0) { $nbrOfFeatures = $#features; }
   elsif ($nbrOfFeatures != $#features and @features != 0) {
      printf STDERR "unexpected number of features: %d (%d)\n",
         $#features+1,$nbrOfFeatures+1;
      exit(1);
   }
   if (@features == 0 or 
       $features[0] eq $boundary) { @features = ($boundary,"O","O"); }
   if (@features < 2) { 
      die "conlleval: unexpected number of features in line $line\n"; 
   }
   if ($raw) {
      if ($features[$#features] eq $oTag) { $features[$#features] = "O"; } 
      if ($features[$#features-1] eq $oTag) { $features[$#features-1] = "O"; } 
      if ($features[$#features] ne "O") { 
         $features[$#features] = "B-$features[$#features]";
      }
      if ($features[$#features-1] ne "O") { 
         $features[$#features-1] = "B-$features[$#features-1]";
      }
   }
   # 20040126 ET code which allows hyphens in the types
   if ($features[$#features] =~ /^([^-]*)-(.*)$/) {
      $guessed = $1;
      $guessedType = $2;
   } else { 
      $guessed = $features[$#features]; 
      $guessedType = ""; 
   }
   pop(@features);
   if ($features[$#features] =~ /^([^-]*)-(.*)$/) {
      $correct = $1;
      $correctType = $2;
   } else { 
      $correct = $features[$#features]; 
      $correctType = ""; 
   }
   pop(@features);
#  ($guessed,$guessedType) = split(/-/,pop(@features));
#  ($correct,$correctType) = split(/-/,pop(@features));
   $guessedType = $guessedType ? $guessedType : "";
   $correctType = $correctType ? $correctType : "";
   $firstItem = shift(@features);

   # 1999-06-26 sentence breaks should always be counted as out of chunk
   if ( $firstItem eq $boundary ) { $guessed = "O"; }

   if ($inCorrect) {
      if ( &endOfChunk($lastCorrect,$correct,$lastCorrectType,$correctType) and
           &endOfChunk($lastGuessed,$guessed,$lastGuessedType,$guessedType) and
           $lastGuessedType eq $lastCorrectType) {
         $inCorrect=$false;
         $correctChunk++;
         $correctChunk{$lastCorrectType} = $correctChunk{$lastCorrectType} ?
             $correctChunk{$lastCorrectType}+1 : 1;
      } elsif ( 
           &endOfChunk($lastCorrect,$correct,$lastCorrectType,$correctType) != 
           &endOfChunk($lastGuessed,$guessed,$lastGuessedType,$guessedType) or
           $guessedType ne $correctType ) {
         $inCorrect=$false; 
      }
   }

   if ( &startOfChunk($lastCorrect,$correct,$lastCorrectType,$correctType) and 
        &startOfChunk($lastGuessed,$guessed,$lastGuessedType,$guessedType) and
        $guessedType eq $correctType) { $inCorrect = $true; }

   if ( &startOfChunk($lastCorrect,$correct,$lastCorrectType,$correctType) ) {
      $foundCorrect++; 
      $foundCorrect{$correctType} = $foundCorrect{$correctType} ?
          $foundCorrect{$correctType}+1 : 1;
   }
   if ( &startOfChunk($lastGuessed,$guessed,$lastGuessedType,$guessedType) ) {
      $foundGuessed++; 
      $foundGuessed{$guessedType} = $foundGuessed{$guessedType} ?
          $foundGuessed{$guessedType}+1 : 1;
   }
   if ( $firstItem ne $boundary ) { 
      if ( $correct eq $guessed and $guessedType eq $correctType ) { 
         $correctTags++; 
      }
      $tokenCounter++; 
   }

   $lastGuessed = $guessed;
   $lastCorrect = $correct;
   $lastGuessedType = $guessedType;
   $lastCorrectType = $correctType;
}
if ($inCorrect) { 
   $correctChunk++;
   $correctChunk{$lastCorrectType} = $correctChunk{$lastCorrectType} ?
       $correctChunk{$lastCorrectType}+1 : 1;
}

if (not $latex) {
   # compute overall precision, recall and FB1 (default values are 0.0)
   $precision = 100*$correctChunk/$foundGuessed if ($foundGuessed > 0);
   $recall = 100*$correctChunk/$foundCorrect if ($foundCorrect > 0);
   $FB1 = 2*$precision*$recall/($precision+$recall)
      if ($precision+$recall > 0);
   
   # print overall performance
   printf "processed $tokenCounter tokens with $foundCorrect phrases; ";
   printf "found: $foundGuessed phrases; correct: $correctChunk.\n";
   if ($tokenCounter>0) {
      printf "accuracy: %6.2f%%; ",100*$correctTags/$tokenCounter;
      printf "precision: %6.2f%%; ",$precision;
      printf "recall: %6.2f%%; ",$recall;
      printf "FB1: %6.2f\n",$FB1;
   }
}

# sort chunk type names
undef($lastType);
@sortedTypes = ();
foreach $i (sort (keys %foundCorrect,keys %foundGuessed)) {
   if (not($lastType) or $lastType ne $i) { 
      push(@sortedTypes,($i));
   }
   $lastType = $i;
}
# print performance per chunk type
if (not $latex) {
   for $i (@sortedTypes) {
      $correctChunk{$i} = $correctChunk{$i} ? $correctChunk{$i} : 0;
      if (not($foundGuessed{$i})) { $foundGuessed{$i} = 0; $precision = 0.0; }
      else { $precision = 100*$correctChunk{$i}/$foundGuessed{$i}; }
      if (not($foundCorrect{$i})) { $recall = 0.0; }
      else { $recall = 100*$correctChunk{$i}/$foundCorrect{$i}; }
      if ($precision+$recall == 0.0) { $FB1 = 0.0; }
      else { $FB1 = 2*$precision*$recall/($precision+$recall); }
      printf "%17s: ",$i;
      printf "precision: %6.2f%%; ",$precision;
      printf "recall: %6.2f%%; ",$recall;
      printf "FB1: %6.2f  %d\n",$FB1,$foundGuessed{$i};
   }
} else {
   print "        & Precision &  Recall  & F\$_{\\beta=1} \\\\\\hline";
   for $i (@sortedTypes) {
      $correctChunk{$i} = $correctChunk{$i} ? $correctChunk{$i} : 0;
      if (not($foundGuessed{$i})) { $precision = 0.0; }
      else { $precision = 100*$correctChunk{$i}/$foundGuessed{$i}; }
      if (not($foundCorrect{$i})) { $recall = 0.0; }
      else { $recall = 100*$correctChunk{$i}/$foundCorrect{$i}; }
      if ($precision+$recall == 0.0) { $FB1 = 0.0; }
      else { $FB1 = 2*$precision*$recall/($precision+$recall); }
      printf "\n%-7s &  %6.2f\\%% & %6.2f\\%% & %6.2f \\\\",
             $i,$precision,$recall,$FB1;
   }
   print "\\hline\n";
   $precision = 0.0;
   $recall = 0;
   $FB1 = 0.0;
   $precision = 100*$correctChunk/$foundGuessed if ($foundGuessed > 0);
   $recall = 100*$correctChunk/$foundCorrect if ($foundCorrect > 0);
   $FB1 = 2*$precision*$recall/($precision+$recall)
      if ($precision+$recall > 0);
   printf "Overall &  %6.2f\\%% & %6.2f\\%% & %6.2f \\\\\\hline\n",
          $precision,$recall,$FB1;
}

exit 0;

# endOfChunk: checks if a chunk ended between the previous and current word
# arguments:  previous and current chunk tags, previous and current types
# note:       this code is capable of handling other chunk representations
#             than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
#             Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006

sub endOfChunk {
   my $prevTag = shift(@_);
   my $tag = shift(@_);
   my $prevType = shift(@_);
   my $type = shift(@_);
   my $chunkEnd = $false;

   if ( $prevTag eq "B" and $tag eq "B" ) { $chunkEnd = $true; }
   if ( $prevTag eq "B" and $tag eq "O" ) { $chunkEnd = $true; }
   if ( $prevTag eq "I" and $tag eq "B" ) { $chunkEnd = $true; }
   if ( $prevTag eq "I" and $tag eq "O" ) { $chunkEnd = $true; }

   if ( $prevTag eq "E" and $tag eq "E" ) { $chunkEnd = $true; }
   if ( $prevTag eq "E" and $tag eq "I" ) { $chunkEnd = $true; }
   if ( $prevTag eq "E" and $tag eq "O" ) { $chunkEnd = $true; }
   if ( $prevTag eq "I" and $tag eq "O" ) { $chunkEnd = $true; }

   if ($prevTag ne "O" and $prevTag ne "." and $prevType ne $type) { 
      $chunkEnd = $true; 
   }

   # corrected 1998-12-22: these chunks are assumed to have length 1
   if ( $prevTag eq "]" ) { $chunkEnd = $true; }
   if ( $prevTag eq "[" ) { $chunkEnd = $true; }

   return($chunkEnd);   
}

# startOfChunk: checks if a chunk started between the previous and current word
# arguments:    previous and current chunk tags, previous and current types
# note:         this code is capable of handling other chunk representations
#               than the default CoNLL-2000 ones, see EACL'99 paper of Tjong
#               Kim Sang and Veenstra http://xxx.lanl.gov/abs/cs.CL/9907006

sub startOfChunk {
   my $prevTag = shift(@_);
   my $tag = shift(@_);
   my $prevType = shift(@_);
   my $type = shift(@_);
   my $chunkStart = $false;

   if ( $prevTag eq "B" and $tag eq "B" ) { $chunkStart = $true; }
   if ( $prevTag eq "I" and $tag eq "B" ) { $chunkStart = $true; }
   if ( $prevTag eq "O" and $tag eq "B" ) { $chunkStart = $true; }
   if ( $prevTag eq "O" and $tag eq "I" ) { $chunkStart = $true; }

   if ( $prevTag eq "E" and $tag eq "E" ) { $chunkStart = $true; }
   if ( $prevTag eq "E" and $tag eq "I" ) { $chunkStart = $true; }
   if ( $prevTag eq "O" and $tag eq "E" ) { $chunkStart = $true; }
   if ( $prevTag eq "O" and $tag eq "I" ) { $chunkStart = $true; }

   if ($tag ne "O" and $tag ne "." and $prevType ne $type) { 
      $chunkStart = $true; 
   }

   # corrected 1998-12-22: these chunks are assumed to have length 1
   if ( $tag eq "[" ) { $chunkStart = $true; }
   if ( $tag eq "]" ) { $chunkStart = $true; }

   return($chunkStart);   
}
