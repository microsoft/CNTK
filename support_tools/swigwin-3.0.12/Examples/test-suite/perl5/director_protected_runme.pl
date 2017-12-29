use strict;
use warnings;
use Test::More tests => 19;
BEGIN { use_ok 'director_protected' }
require_ok 'director_protected';

{
  package FooBar;
  use base 'director_protected::Bar';
  sub ping { 'FooBar::ping();' }
}
{
  package FooBar2;
  use base 'director_protected::Bar';
  sub ping { 'FooBar2::ping();' }
  sub pang { 'FooBar2::pang();' }
}

my $b  = director_protected::Bar->new();
isa_ok $b, 'director_protected::Bar';
my $f  = $b->create();
my $fb = FooBar->new();
isa_ok $fb, 'FooBar';
my $fb2 = FooBar2->new();
isa_ok $fb2, 'FooBar2';

is $b->used(), "Foo::pang();Bar::pong();Foo::pong();Bar::ping();";
eval { $f->used() };
like $@, qr/protected member/;
is $fb->used(), "Foo::pang();Bar::pong();Foo::pong();FooBar::ping();";
is $fb2->used(), "FooBar2::pang();Bar::pong();Foo::pong();FooBar2::ping();";

is $b->pong(), "Bar::pong();Foo::pong();Bar::ping();";
is $f->pong(), "Bar::pong();Foo::pong();Bar::ping();";
is $fb->pong(), "Bar::pong();Foo::pong();FooBar::ping();";
is $fb2->pong(), "Bar::pong();Foo::pong();FooBar2::ping();";

eval { $b->ping() };
like $@, qr/protected member/;
eval { $f->ping () };
like $@, qr/protected member/;
is $fb->ping(), 'FooBar::ping();';
is $fb2->ping(), 'FooBar2::ping();';

eval { $b->pang() };
like $@, qr/protected member/;
eval { $f->pang() };
like $@, qr/protected member/;
