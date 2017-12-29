use strict;
use warnings;
use Test::More tests => 3;
BEGIN { use_ok('director_alternating') }
require_ok('director_alternating');

my $id = director_alternating::getBar()->id();
is($id, director_alternating::idFromGetBar(), "got Bar id");
