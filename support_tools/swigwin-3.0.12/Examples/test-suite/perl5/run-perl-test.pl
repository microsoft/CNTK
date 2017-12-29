#!/usr/bin/perl
    eval 'exec /usr/bin/perl -S $0 ${1+"$@"}'
	if $running_under_some_shell;
#!/usr/bin/perl -w

use strict;

my $command = shift @ARGV;

my $output = `$^X $command 2>&1`;

die "SWIG Perl test failed: \n\n$output\n"
  if $?;

exit(0);
