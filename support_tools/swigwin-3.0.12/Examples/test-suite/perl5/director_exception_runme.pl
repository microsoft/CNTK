use strict;
use warnings;
use Test::More tests => 7;
BEGIN { use_ok 'director_exception' }
require_ok 'director_exception';

{
  package MyFoo;
  use base 'director_exception::Foo';
  sub ping {
    die "MyFoo::ping() EXCEPTION";
  }
}
{
  package MyFoo2;
  use base 'director_exception::Foo';
  sub ping {
    # error should return a string
    return bless [ 1 ], 'main';
  }
}
{
  package MyFoo3;
  use base 'director_exception::Foo';
  sub ping {
    # error should return a string
    return sub { 1 }
  }
}

{
  my $a = MyFoo->new();
  my $b = director_exception::launder($a);
  eval { $b->pong() };
  like($@, qr/\bMyFoo::ping\(\) EXCEPTION\b/,
      'MyFoo.pong() error content preserved');
}
{
  my $a = MyFoo2->new();
  my $b = director_exception::launder($a);
  eval { $b->pong() };
  like($@, qr/\bTypeError\b/,
      'MyFoo2.pong() error content preserved');
}
{
  my $a = MyFoo3->new();
  my $b = director_exception::launder($a);
  eval { $b->pong() };
  like($@, qr/\bTypeError\b/,
      'MyFoo2.pong() error content preserved');
}

eval { die director_exception::Exception1->new() };
isa_ok($@, 'director_exception::Exception1', 'Exception1');

eval { die director_exception::Exception2->new() };
isa_ok($@, 'director_exception::Exception2', 'Exception2');
