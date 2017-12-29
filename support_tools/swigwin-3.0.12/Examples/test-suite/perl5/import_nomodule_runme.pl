use strict;
use warnings;
use Test::More tests => 4;
BEGIN { use_ok('import_nomodule') }
require_ok('import_nomodule');

my $f = import_nomodule::create_Foo();
import_nomodule::test1($f,42);
ok(1, "basecase");

my $b = new import_nomodule::Bar();
import_nomodule::test1($b,37);
ok(1, "testcase");
