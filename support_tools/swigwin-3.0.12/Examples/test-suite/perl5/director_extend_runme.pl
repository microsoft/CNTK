use strict;
use warnings;
use Test::More tests => 5;
BEGIN { use_ok 'director_extend' }
require_ok 'director_extend';

{
  package MyObject;
  use base 'director_extend::SpObject';
  sub getFoo { 123 }
}

my $m = MyObject->new();
isa_ok $m, 'MyObject';
is($m->dummy(), 666, '1st call');
is($m->dummy(), 666, '2nd call');
