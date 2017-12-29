use strict;
use warnings;
use Test::More tests => 4;
BEGIN { use_ok('li_cdata_carrays') }
require_ok('li_cdata_carrays');

my $ia = li_cdata_carrays::intArray->new(5);
for (0..4) {
  $ia->setitem($_, $_**2);
}
ok(1, "setitems");
my $x = pack q{I5}, map $_**2, (0..4);
my $y = li_cdata_carrays::cdata_int($ia->cast, 5);
is($x, $y, "carrays");
