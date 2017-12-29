use strict;
use warnings;
use Test::More tests => 9;
BEGIN { use_ok 'director_constructor' }
require_ok 'director_constructor';

{
  package Test;
  use base 'director_constructor::Foo';
  sub doubleit { my($self) = @_;
    $self->{a} *= 2;
  }
  sub test { 3 }
}
my $t = Test->new(5);
isa_ok $t, 'Test';
is $t->getit, 5;
is $t->do_test, 3;

$t->doubleit();

is $t->getit, 10;

{
  package Wrong;
  use base 'director_constructor::Foo';
  sub doubleit { my($self) = @_;
  # calling this should trigger a type error on attribute
  # assignment
    $self->{a} = {};
  }
  sub test {
    # if c++ calls this, retval copyout should trigger a type error
    return bless {}, 'TotallyBogus';
  }
}

# TODO: these TypeErrors in director classes should be more detailed
my $w = Wrong->new(12);
is eval { $w->doubleit() }, undef;
like $@, qr/TypeError/;
is $w->getit(), 12, 'W.a should be unaffected';

# TODO: this is giving an unhandled C++ exception right now
#is eval { $W->do_test() }, undef;
#like $@, qr/TypeError/;
