use strict;

my $inmlf = $ARGV[0];
my $outmlf = $ARGV[1];

open(IN,"<$inmlf") or die "cannot open $inmlf: $%!";
open(OUT,">$outmlf") or die "cannot open $outmlf: $%!";

my $line;
my $dr;
my ($beg,$end,$sym,$id);
print OUT "#!MLF!#\n";

foreach (<IN>){
    chomp;
    s,^\s+,,;
    s,\s+$,,;
    if (/^\.$/){
	($beg,$end,$sym,$id) =split(/\s+/,$line);
	my $id = $dr-1;
	print OUT "0 $end dr${dr} $id\n.\n"
    }
    elsif (/^\"/){ #match utterance name, find dr
	print OUT "$_\n";
	/\-dr(\d)\-/;
	$dr = $1;
#	print "$_\t$1\n";
    }
    else{
	$line=$_;
    }
}
close(IN);
close(OUT);
