use strict;
use warnings;
use Test::More tests => 10;
BEGIN { use_ok('char_binary') }
require_ok('char_binary');

my $t = char_binary::Test->new();

is($t->strlen('hile'), 4, "string typemap");
is($t->ustrlen('hile'), 4, "unsigned string typemap");

is($t->strlen("hil\0"), 4, "string typemap");
is($t->ustrlen("hil\0"), 4, "unsigned string typemap");

#
# creating a raw char*
#
my $pc = char_binary::new_pchar(5);
char_binary::pchar_setitem($pc, 0, 'h');
char_binary::pchar_setitem($pc, 1, 'o');
char_binary::pchar_setitem($pc, 2, 'l');
char_binary::pchar_setitem($pc, 3, 'a');
char_binary::pchar_setitem($pc, 4, 0);


is($t->strlen($pc), 4, "string typemap");
is($t->ustrlen($pc), 4, "unsigned string typemap");

$char_binary::var_pchar = $pc;
is($char_binary::var_pchar, "hola", "pointer case");

$char_binary::var_namet = $pc;
is($char_binary::var_namet, "hola", "pointer case");

char_binary::delete_pchar($pc);
