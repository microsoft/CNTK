use strict;
use warnings;
use Test::More tests => 6;
BEGIN { use_ok 'director_default' }
require_ok 'director_default';

my $f;

$f = director_default::Foo->new();
isa_ok $f, 'director_default::Foo';
$f = director_default::Foo->new(1);
isa_ok $f, 'director_default::Foo';


$f = director_default::Bar->new();
isa_ok $f, 'director_default::Bar';
$f = director_default::Bar->new(1);
isa_ok $f, 'director_default::Bar';
