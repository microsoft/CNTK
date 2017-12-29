use strict;
use warnings;
use Test::More tests => 14;
BEGIN { use_ok('primitive_ref') }
require_ok('primitive_ref');

is(primitive_ref::ref_int(3), 3, "ref_int");
is(primitive_ref::ref_uint(3), 3, "ref_uint");
is(primitive_ref::ref_short(3), 3, "ref_short");
is(primitive_ref::ref_ushort(3), 3, "ref_ushort");
is(primitive_ref::ref_long(3), 3, "ref_long");
is(primitive_ref::ref_ulong(3), 3, "ref_ulong");
is(primitive_ref::ref_schar(3), 3, "ref_schar");
is(primitive_ref::ref_uchar(3), 3, "ref_uchar");
is(primitive_ref::ref_bool(1), 1, "ref_bool");
is(primitive_ref::ref_float(3.5), 3.5, "ref_float");
is(primitive_ref::ref_double(3.5), 3.5, "ref_double");
is(primitive_ref::ref_char('x'), 'x', "ref_char");
