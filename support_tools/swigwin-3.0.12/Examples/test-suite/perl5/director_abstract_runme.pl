use strict;
use warnings;
use Test::More tests => 13;
BEGIN { use_ok('director_abstract') }
require_ok('director_abstract');

{
  package MyFoo;
  use base 'director_abstract::Foo';
  sub ping {
    return 'MyFoo::ping()';
  }
}

my $f = MyFoo->new();

is($f->ping, "MyFoo::ping()");

is($f->pong(),"Foo::pong();MyFoo::ping()");

{
  package MyExample1;
  use base 'director_abstract::Example1';
  sub Color { my($self, $r, $g, $b) = @_;
    return $r;
  }
}
{
  package MyExample2;
  use base 'director_abstract::Example2';
  sub Color { my($self, $r, $g, $b) = @_;
    return $g;
  }
}
{
  package MyExample3;
  use base 'director_abstract::Example3_i';
  sub Color { my($self, $r, $g, $b) = @_;
    return $b;
  }
}

my $me1 = MyExample1->new();
isa_ok($me1, 'MyExample1');
is(director_abstract::Example1::get_color($me1, 1, 2, 3), 1, 'me1');

my $me2 = MyExample2->new(1,2);
isa_ok($me2, 'MyExample2');
is(director_abstract::Example2::get_color($me2, 1, 2, 3), 2, 'me2');

my $me3 = MyExample3->new();
isa_ok($me3, 'MyExample3');
is(director_abstract::Example3_i::get_color($me3, 1, 2, 3), 3, 'me3');

eval { $me1 = director_abstract::Example1->new() };
like($@, qr/\babstract\b/i, 'E1.new()');

eval { $me2 = director_abstract::Example2->new() };
like($@, qr/Example2/, 'E2.new()');

eval { $me3 = director_abstract::Example3_i->new() };
like($@, qr/\babstract\b/i, 'E3.new()');
