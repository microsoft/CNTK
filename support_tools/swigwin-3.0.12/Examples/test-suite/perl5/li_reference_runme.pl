use strict;
use warnings;
use Test::More tests => 68;
BEGIN { use_ok('li_reference') }
require_ok('li_reference');

sub chk { my($type, $call, $v1, $v2) = @_;
    $li_reference::FrVal = $v1;
    my $v = $v2;
    eval { $call->(\$v) };
    is($@, '', "$type check");
    is($li_reference::ToVal, $v2, "$type out");
    is($v, $v1, "$type in");
}
chk("double*", \&li_reference::PDouble, 12.2, 18.6);
chk("double&", \&li_reference::RDouble, 32.5, 64.8);
chk("float*",  \&li_reference::PFloat,  64.5, 96.0);
chk("float&",  \&li_reference::RFloat,  98.5, 6.25);
chk("int*",    \&li_reference::PInt,    1887, 3356);
chk("int&",    \&li_reference::RInt,    2622, 9867);
chk("short*",  \&li_reference::PShort,  4752, 3254);
chk("short&",  \&li_reference::RShort,  1898, 5757);
chk("long*",   \&li_reference::PLong,   6687, 7132);
chk("long&",   \&li_reference::RLong,   8346, 4398);
chk("uint*",   \&li_reference::PUInt,   6853, 5529);
chk("uint&",   \&li_reference::RUInt,   5483, 7135);
chk("ushort*", \&li_reference::PUShort, 9960, 9930);
chk("ushort&", \&li_reference::RUShort, 1193, 4178);
chk("ulong*",  \&li_reference::PULong,  7960, 4788);
chk("ulong&",  \&li_reference::RULong,  8829, 1603);
chk("uchar*",  \&li_reference::PUChar,  110,  239);
chk("uchar&",  \&li_reference::RUChar,  15,   97);
chk("char*",   \&li_reference::PChar,   -7,   118);
chk("char&",   \&li_reference::RChar,   -3,  -107);
chk("bool*",   \&li_reference::PBool,   0,    1);
chk("bool&",   \&li_reference::RBool,   1,    0);
