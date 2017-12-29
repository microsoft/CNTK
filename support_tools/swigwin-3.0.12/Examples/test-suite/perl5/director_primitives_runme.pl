use strict;
use warnings;
use Test::More tests => 27;
BEGIN { use_ok 'director_primitives' }
require_ok 'director_primitives';

{
  package PerlDerived;
  use base 'director_primitives::Base';
  sub NoParmsMethod {
  }
  sub BoolMethod { my($self, $x) = @_;
    return $x;
  }
  sub IntMethod { my($self, $x) = @_;
    return $x;
  }
  sub UIntMethod { my($self, $x) = @_;
    return $x;
  }
  sub FloatMethod { my($self, $x) = @_;
    return $x;
  }
  sub CharPtrMethod { my($self, $x) = @_;
    return $x;
  }
  sub ConstCharPtrMethod { my($self, $x) = @_;
    return $x;
  }
  sub EnumMethod { my($self, $x) = @_;
    return $x;
  }
  sub ManyParmsMethod {
  }
}

my $myCaller = director_primitives::Caller->new();
isa_ok $myCaller, 'director_primitives::Caller';

{
  my $myBase = director_primitives::Base->new(100.0);
  makeCalls($myCaller, $myBase);
}
{
  my $myBase = director_primitives::Derived->new(200.0);
  makeCalls($myCaller, $myBase);
}
{
  my $myBase = PerlDerived->new(300.0);
  makeCalls($myCaller, $myBase);
}

sub makeCalls { my($myCaller, $myBase) = @_;
  $myCaller->set($myBase);
  $myCaller->NoParmsMethodCall();
  is $myCaller->BoolMethodCall(1), '1';
  is $myCaller->BoolMethodCall(0), '';
  is $myCaller->IntMethodCall(-123), -123;
  is $myCaller->UIntMethodCall(123), 123;
  is $myCaller->FloatMethodCall(-123 / 128), -0.9609375;
  is $myCaller->CharPtrMethodCall("test string"), "test string";
  is $myCaller->ConstCharPtrMethodCall("another string"), "another string";
  is $myCaller->EnumMethodCall($director_primitives::HShadowHard), $director_primitives::HShadowHard;
  $myCaller->ManyParmsMethodCall(1, -123, 123, 123.456, "test string", "another string", $director_primitives::HShadowHard);
  $myCaller->NotOverriddenMethodCall();
  $myCaller->reset();
}

