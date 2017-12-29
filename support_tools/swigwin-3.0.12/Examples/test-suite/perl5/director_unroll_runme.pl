use strict;
use warnings;
use Test::More tests => 3;
BEGIN { use_ok 'director_unroll' };
require_ok 'director_unroll';

{
  package MyFoo;
  use base 'director_unroll::Foo';
  sub ping { "MyFoo::ping()" }
}

$a = MyFoo->new();
$b = director_unroll::Bar->new();
$b->set($a);
my $c = $b->get();
is(${$a->this}, ${$c->this}, "unrolling");
