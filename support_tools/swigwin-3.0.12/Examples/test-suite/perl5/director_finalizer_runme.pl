use strict;
use warnings;
use Test::More tests => 13;
BEGIN { use_ok('director_finalizer') }
require_ok('director_finalizer');

{
  package MyFoo;
  use base 'director_finalizer::Foo';
  sub DESTROY { my($self, $final) = @_;
    $self->orStatus(2) if $final;
    shift->SUPER::DESTROY(@_);
  }
}

{
  director_finalizer::resetStatus();
  my $f = MyFoo->new();
  undef $f;
  is(director_finalizer::getStatus(), 3, 'shadow release fires destructor');
}

{ # again, this time with DESTROY
  director_finalizer::resetStatus();
  my $f = MyFoo->new();
  $f->DESTROY();
  is(director_finalizer::getStatus(), 3, 'DESTROY method fires destructor');
}

{
  director_finalizer::resetStatus();
  my $f = MyFoo->new();
  director_finalizer::launder($f);
  is(director_finalizer::getStatus(), 0, 'wrap release does not fire destructor');
  undef $f;
  is(director_finalizer::getStatus(), 3, 'shadow release still fires destructor');
}

{ # again, this time with DESTROY
  director_finalizer::resetStatus();
  my $f = MyFoo->new();
  director_finalizer::launder($f);
  is(director_finalizer::getStatus(), 0, 'wrap release does not fire destructor');
  $f->DESTROY();
  is(director_finalizer::getStatus(), 3, 'DESTROY method still fires destructor');
}

{
  director_finalizer::resetStatus();
  my $f = MyFoo->new();
  $f->DISOWN();
  is(director_finalizer::getStatus(), 0, 'shadow release does not fire destructor of disowned object');
  director_finalizer::deleteFoo($f);
  is(director_finalizer::getStatus(), 3, 'c++ release fires destructors of disowned object');
}

{ # again, this time with DESTROY
  my $f = MyFoo->new();
  $f->DISOWN();
  director_finalizer::deleteFoo($f);
  director_finalizer::resetStatus();
  $f->DESTROY();
  is(director_finalizer::getStatus(), 0, 'DESTROY method does not fire destructor of disowned object');
}

{
  director_finalizer::resetStatus();
  my $f = MyFoo->new();
  $f->DISOWN();
  my $g = director_finalizer::launder($f);
  undef $f;
  director_finalizer::deleteFoo($g);
  is(director_finalizer::getStatus(), 3, 'c++ release fires destructors on disowned opaque object');
}

{ # again, this time with DESTROY
  director_finalizer::resetStatus();
  my $f = MyFoo->new();
  $f->DISOWN();
  my $g = director_finalizer::launder($f);
  $f->DESTROY();
  director_finalizer::deleteFoo($g);
  is(director_finalizer::getStatus(), 3, 'c++ release fires destructors on disowned opaque object after DESTROY');
}
