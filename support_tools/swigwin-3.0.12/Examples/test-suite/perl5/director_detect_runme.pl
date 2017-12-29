use strict;
use warnings;
use Test::More tests => 9;
BEGIN { use_ok 'director_detect' }
require_ok 'director_detect';

{
  package MyBar;
  use base 'director_detect::Bar';
  sub new { my $class = shift;
    my $val = @_ ? shift : 2;
    my $self = $class->SUPER::new();
    $self->{val} = $val;
    return $self;
  }
  sub get_value { my($self) = @_;
    $self->{val}++;
    return $self->{val};
  }
  sub get_class { my($self) = @_;
    $self->{val}++;
    return director_detect::A->new();
  }
  sub just_do_it { my($self) = @_;
    $self->{val}++;
  }
  sub clone { my($self) = @_;
    MyBar->new($self->{val});
  }
}

my $b = MyBar->new();
isa_ok $b, 'MyBar';

my $f = $b->baseclass();
isa_ok $f, 'director_detect::Foo';
is $f->get_value(), 3;

isa_ok $f->get_class(), 'director_detect::A';
$f->just_do_it();

my $c = $b->clone();
isa_ok $c, 'MyBar';
is $b->{val}, 5;
is $c->get_value(), 6;
