use strict;
use warnings;
use Test::More tests => 6;
BEGIN { use_ok 'director_ignore' }
require_ok 'director_ignore';

{
  package DIgnoresDerived;
  use base 'director_ignore::DIgnores';
  sub PublicMethod1 {
    return 18.75;
  }
}
{
  package DAbstractIgnoresDerived;
  use base 'director_ignore::DAbstractIgnores';
}

my $a = DIgnoresDerived->new();
isa_ok $a, 'DIgnoresDerived';
is $a->Triple(5), 15;

my $b = DAbstractIgnoresDerived->new();
isa_ok $b, 'DAbstractIgnoresDerived';
is $b->Quadruple(5), 20;
