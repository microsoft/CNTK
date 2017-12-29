use strict;
use warnings;
use Test::More tests => 29;
BEGIN { use_ok 'director_classes' }
require_ok 'director_classes';

{
  package PerlDerived;
  use base 'director_classes::Base';
  sub Val { $_[1] }
  sub Ref { $_[1] }
  sub Ptr { $_[1] }
  sub FullyOverloaded {
    my $rv = shift->SUPER::FullyOverloaded(@_);
    $rv =~ s/Base/__PACKAGE__/sge;
    return $rv;
  }
  sub SemiOverloaded {
    # this is going to be awkward because we can't really
    # semi-overload in Perl, but we can sort of fake it.
    return shift->SUPER::SemiOverloaded(@_) unless $_[0] =~ /^\d+/;
    my $rv = shift->SUPER::SemiOverloaded(@_);
    $rv =~ s/Base/__PACKAGE__/sge;
    return $rv;
  }
  sub DefaultParms {
    my $rv = shift->SUPER::DefaultParms(@_);
    $rv =~ s/Base/__PACKAGE__/sge;
    return $rv;
  }
}

{
  my $c = director_classes::Caller->new();
  makeCalls($c, director_classes::Base->new(100.0));
  makeCalls($c, director_classes::Derived->new(200.0));
  makeCalls($c, PerlDerived->new(300.0));
}

sub makeCalls { my($caller, $base) = @_;
  my $bname = ref $base;
  $bname = $1 if $bname =~ /^director_classes::(.*)$/;
  $caller->set($base);
  my $dh = director_classes::DoubleHolder->new(444.555);
  is($caller->ValCall($dh)->{val}, $dh->{val}, "$bname.Val");
  is($caller->RefCall($dh)->{val}, $dh->{val}, "$bname.Ref");
  is($caller->PtrCall($dh)->{val}, $dh->{val}, "$bname.Ptr");
  is($caller->FullyOverloadedCall(1),
      "${bname}::FullyOverloaded(int)",
      "$bname.FullyOverloaded(int)");
  is($caller->FullyOverloadedCall(''),
      "${bname}::FullyOverloaded(bool)",
      "$bname.FullyOverloaded(bool)");
TODO: {
  local $TODO = 'investigation needed here' if $bname eq 'PerlDerived';
  is($caller->SemiOverloadedCall(-678),
      "${bname}::SemiOverloaded(int)",
      "$bname.SemiOverloaded(int)");
}
  is($caller->SemiOverloadedCall(''),
      "Base::SemiOverloaded(bool)",
      "$bname.SemiOverloaded(bool)");
  is($caller->DefaultParmsCall(10, 2.2),
      "${bname}::DefaultParms(int, double)",
      "$bname.DefaultParms(int, double)");
  is($caller->DefaultParmsCall(10),
      "${bname}::DefaultParms(int)",
      "$bname.DefaultParms(int)");
  $caller->reset();
}
