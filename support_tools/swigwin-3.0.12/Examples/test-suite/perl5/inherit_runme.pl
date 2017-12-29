use strict;
use warnings;
use Test::More tests => 3;
BEGIN { use_ok('inherit') }
require_ok('inherit');

my $der = new inherit::CDerived();
is($der->Foo(), "CBase::Foo", "inherit test");

