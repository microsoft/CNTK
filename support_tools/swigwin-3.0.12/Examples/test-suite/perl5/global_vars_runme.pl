use strict;
use warnings;
use Test::More tests => 4;
BEGIN { use_ok('global_vars') }
require_ok('global_vars');

my $an = new global_vars::A();
isa_ok($an, 'global_vars::A');
$global_vars::ap = $an;
is($global_vars::ap, $an, "global var assignment");

