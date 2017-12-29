use strict;
use warnings;
use Test::More tests => 5;
BEGIN { use_ok 'director_enum' }
require_ok 'director_enum';

{
  package MyFoo;
  use base 'director_enum::Foo';
  sub say_hi { my($self, $val) = @_;
    return $val;
  }
}

my $b = director_enum::Foo->new();
isa_ok $b, 'director_enum::Foo';
my $a = MyFoo->new();
isa_ok $a, 'MyFoo';

is $a->say_hi($director_enum::hello),
   $a->say_hello($director_enum::hi);
