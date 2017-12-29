use strict;
use warnings;
use Test::More tests => 12;
BEGIN { use_ok 'director_basic' }
require_ok 'director_basic';

{
  package MyFoo;
  use base 'director_basic::Foo';
  sub ping {
    return 'MyFoo::ping()';
  }
}

{
  package MyOverriddenClass;
  use base 'director_basic::MyClass';
  use fields qw(expectNull nonNullReceived);
  sub new {
    my $self = shift->SUPER::new(@_);
    $self->{expectNull} = undef;
    $self->{nonNullReceived} = undef;
    return $self;
  }
  sub pmethod { my($self, $b) = @_;
    die "null not received as expected"
      if $self->{expectNull} and defined $b;
    return $b;
  }
}

{
  my $a = MyFoo->new();
  isa_ok $a, 'MyFoo';
  is $a->ping(), 'MyFoo::ping()', 'a.ping()';
  is $a->pong(), 'Foo::pong();MyFoo::ping()', 'a.pong()';

  my $b = director_basic::Foo->new();
  isa_ok $b, 'director_basic::Foo';
  is $b->ping(), 'Foo::ping()', 'b.ping()';
  is $b->pong(), 'Foo::pong();Foo::ping()', 'b.pong()';

  my $a1 = director_basic::A1->new(1, undef);
  isa_ok $a1, 'director_basic::A1';
  is $a1->rg(2), 2, 'A1.rg';

  my $my = MyOverriddenClass->new();
  $my->{expectNull} = 1;
  is(director_basic::MyClass::call_pmethod($my, undef), undef,
      'null pointer marshalling');

  my $myBar = director_basic::Bar->new();
  $my->{expectNull} = undef;
  my $myNewBar = director_basic::MyClass::call_pmethod($my, $myBar);
  isnt($myNewBar, undef, 'non-null pointer marshalling');
  $myNewBar->{x} = 10;
}
