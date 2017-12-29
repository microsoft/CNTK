# an adaptation of ../java/enum_thorough_runme.java
use strict;
use warnings;
use Test::More tests => 298;
BEGIN { use_ok('enum_thorough') }
require_ok('enum_thorough');

is($enum_thorough::ReallyAnInteger, 200, "Test Anon 1");

{
	my $red = $enum_thorough::red;
	is(enum_thorough::colourTest1($red), $red, "colourTest1");
	is(enum_thorough::colourTest2($red), $red, "colourTest2");
	is(enum_thorough::colourTest3($red), $red, "colourTest3");
	is(enum_thorough::colourTest4($red), $red, "colourTest4");
	isnt($enum_thorough::myColour, $red, "colour global get");
	$enum_thorough::myColour = $red;
	is($enum_thorough::myColour, $red, "colour global set");
}
{
	my $s = enum_thorough::SpeedClass->new();
	my $speed = $enum_thorough::SpeedClass::slow;
	ok(defined($speed), "SpeedClass.slow");
	is($s->speedTest1($speed), $speed, "speedTest 1");
	is($s->speedTest2($speed), $speed, "speedTest 2");
	is($s->speedTest3($speed), $speed, "speedTest 3");
	is($s->speedTest4($speed), $speed, "speedTest 4");
	is($s->speedTest5($speed), $speed, "speedTest 5");
	is($s->speedTest6($speed), $speed, "speedTest 6");
	is($s->speedTest7($speed), $speed, "speedTest 7");
	is($s->speedTest8($speed), $speed, "speedTest 8");
	is(enum_thorough::speedTest1($speed), $speed, "speedTest Global 1");
	is(enum_thorough::speedTest2($speed), $speed, "speedTest Global 2");
	is(enum_thorough::speedTest3($speed), $speed, "speedTest Global 3");
	is(enum_thorough::speedTest4($speed), $speed, "speedTest Global 4");
	is(enum_thorough::speedTest5($speed), $speed, "speedTest Global 5");
}
{
	my $s = enum_thorough::SpeedClass->new();
	my $slow = $enum_thorough::SpeedClass::slow;
	my $lightning = $enum_thorough::SpeedClass::lightning;
	is($s->{mySpeedtd1}, $slow, "mySpeedtd1 1");
	is($s->{mySpeedtd1}, 10, "mySpeedtd1 2");
	$s->{mySpeedtd1} = $lightning;
	is($s->{mySpeedtd1}, $lightning, "mySpeedtd1 3");
	is($s->{mySpeedtd1}, 31, "mySpeedtd1 4");
}
is(enum_thorough::namedanonTest1($enum_thorough::NamedAnon2),
	$enum_thorough::NamedAnon2, "namedanonTest1");
{
	my $val = $enum_thorough::TwoNames2;
	is(enum_thorough::twonamesTest1($val), $val, "twonamesTest 1");
	is(enum_thorough::twonamesTest2($val), $val, "twonamesTest 2");
	is(enum_thorough::twonamesTest3($val), $val, "twonamesTest 3");
}
{ local $TODO = "shouldn't namespaces drop into a package?";
	my $val = $enum_thorough::AnonSpace::NamedAnonSpace2;
	ok(defined($val), "found enum value");
SKIP: {
	skip "enum value not in expected package", 3 unless defined $val;
	is(enum_thorough::namedanonspaceTest1($val), $val, "namedanonspaceTest 1");
	is(enum_thorough::namedanonspaceTest2($val), $val, "namedanonspaceTest 2");
	is(enum_thorough::namedanonspaceTest3($val), $val, "namedanonspaceTest 3");
}}
{
	my $t = enum_thorough::TemplateClassInt->new();
	my $galileo = $enum_thorough::TemplateClassInt::galileo;
	is($t->scientistsTest1($galileo), $galileo, "scientistsTest 1");
	is($t->scientistsTest2($galileo), $galileo, "scientistsTest 2");
	is($t->scientistsTest3($galileo), $galileo, "scientistsTest 3");
	is($t->scientistsTest4($galileo), $galileo, "scientistsTest 4");
	is($t->scientistsTest5($galileo), $galileo, "scientistsTest 5");
	is($t->scientistsTest6($galileo), $galileo, "scientistsTest 6");
	is($t->scientistsTest7($galileo), $galileo, "scientistsTest 7");
	is($t->scientistsTest8($galileo), $galileo, "scientistsTest 8");
	is($t->scientistsTest9($galileo), $galileo, "scientistsTest 9");
	is($t->scientistsTestB($galileo), $galileo, "scientistsTest B");
	is($t->scientistsTestD($galileo), $galileo, "scientistsTest D");
	is($t->scientistsTestE($galileo), $galileo, "scientistsTest E");
	is($t->scientistsTestF($galileo), $galileo, "scientistsTest F");
	is($t->scientistsTestG($galileo), $galileo, "scientistsTest G");
	is($t->scientistsTestH($galileo), $galileo, "scientistsTest H");
	is($t->scientistsTestI($galileo), $galileo, "scientistsTest I");
	is($t->scientistsTestJ($galileo), $galileo, "scientistsTest J");

	is(enum_thorough::scientistsTest1($galileo), $galileo, "scientistsTest Global 1");
	is(enum_thorough::scientistsTest2($galileo), $galileo, "scientistsTest Global 2");
	is(enum_thorough::scientistsTest3($galileo), $galileo, "scientistsTest Global 3");
	is(enum_thorough::scientistsTest4($galileo), $galileo, "scientistsTest Global 4");
	is(enum_thorough::scientistsTest5($galileo), $galileo, "scientistsTest Global 5");
	is(enum_thorough::scientistsTest6($galileo), $galileo, "scientistsTest Global 6");
	is(enum_thorough::scientistsTest7($galileo), $galileo, "scientistsTest Global 7");
	is(enum_thorough::scientistsTest8($galileo), $galileo, "scientistsTest Global 8");
}
{
	my $t = enum_thorough::TClassInt->new();
	my $bell = $enum_thorough::TClassInt::bell;
	my $galileo = $enum_thorough::TemplateClassInt::galileo;
	is($t->scientistsNameTest1($bell), $bell, "scientistsNameTest 1");
	is($t->scientistsNameTest2($bell), $bell, "scientistsNameTest 2");
	is($t->scientistsNameTest3($bell), $bell, "scientistsNameTest 3");
	is($t->scientistsNameTest4($bell), $bell, "scientistsNameTest 4");
	is($t->scientistsNameTest5($bell), $bell, "scientistsNameTest 5");
	is($t->scientistsNameTest6($bell), $bell, "scientistsNameTest 6");
	is($t->scientistsNameTest7($bell), $bell, "scientistsNameTest 7");
	is($t->scientistsNameTest8($bell), $bell, "scientistsNameTest 8");
	is($t->scientistsNameTest9($bell), $bell, "scientistsNameTest 9");
	is($t->scientistsNameTestB($bell), $bell, "scientistsNameTest B");
	is($t->scientistsNameTestD($bell), $bell, "scientistsNameTest D");
	is($t->scientistsNameTestE($bell), $bell, "scientistsNameTest E");
	is($t->scientistsNameTestF($bell), $bell, "scientistsNameTest F");
	is($t->scientistsNameTestG($bell), $bell, "scientistsNameTest G");
	is($t->scientistsNameTestH($bell), $bell, "scientistsNameTest H");
	is($t->scientistsNameTestI($bell), $bell, "scientistsNameTest I");

	is($t->scientistsNameSpaceTest1($bell), $bell, "scientistsNameSpaceTest 1");
	is($t->scientistsNameSpaceTest2($bell), $bell, "scientistsNameSpaceTest 2");
	is($t->scientistsNameSpaceTest3($bell), $bell, "scientistsNameSpaceTest 3");
	is($t->scientistsNameSpaceTest4($bell), $bell, "scientistsNameSpaceTest 4");
	is($t->scientistsNameSpaceTest5($bell), $bell, "scientistsNameSpaceTest 5");
	is($t->scientistsNameSpaceTest6($bell), $bell, "scientistsNameSpaceTest 6");
	is($t->scientistsNameSpaceTest7($bell), $bell, "scientistsNameSpaceTest 7");

	is($t->scientistsOtherTest1($galileo), $galileo, "scientistsOtherTest 1");
	is($t->scientistsOtherTest2($galileo), $galileo, "scientistsOtherTest 2");
	is($t->scientistsOtherTest3($galileo), $galileo, "scientistsOtherTest 3");
	is($t->scientistsOtherTest4($galileo), $galileo, "scientistsOtherTest 4");
	is($t->scientistsOtherTest5($galileo), $galileo, "scientistsOtherTest 5");
	is($t->scientistsOtherTest6($galileo), $galileo, "scientistsOtherTest 6");
	is($t->scientistsOtherTest7($galileo), $galileo, "scientistsOtherTest 7");

	is(enum_thorough::scientistsNameTest1($bell), $bell, "scientistsNameTest Global 1");
	is(enum_thorough::scientistsNameTest2($bell), $bell, "scientistsNameTest Global 2");
	is(enum_thorough::scientistsNameTest3($bell), $bell, "scientistsNameTest Global 3");
	is(enum_thorough::scientistsNameTest4($bell), $bell, "scientistsNameTest Global 4");
	is(enum_thorough::scientistsNameTest5($bell), $bell, "scientistsNameTest Global 5");
	is(enum_thorough::scientistsNameTest6($bell), $bell, "scientistsNameTest Global 6");
	is(enum_thorough::scientistsNameTest7($bell), $bell, "scientistsNameTest Global 7");

	is(enum_thorough::scientistsNameSpaceTest1($bell), $bell, "scientistsNameSpaceTest Global 1");
	is(enum_thorough::scientistsNameSpaceTest2($bell), $bell, "scientistsNameSpaceTest Global 2");
	is(enum_thorough::scientistsNameSpaceTest3($bell), $bell, "scientistsNameSpaceTest Global 3");
	is(enum_thorough::scientistsNameSpaceTest4($bell), $bell, "scientistsNameSpaceTest Global 4");
	is(enum_thorough::scientistsNameSpaceTest5($bell), $bell, "scientistsNameSpaceTest Global 5");
	is(enum_thorough::scientistsNameSpaceTest6($bell), $bell, "scientistsNameSpaceTest Global 6");
	is(enum_thorough::scientistsNameSpaceTest7($bell), $bell, "scientistsNameSpaceTest Global 7");

	is(enum_thorough::scientistsNameSpaceTest8($bell), $bell, "scientistsNameSpaceTest Global 8");
	is(enum_thorough::scientistsNameSpaceTest9($bell), $bell, "scientistsNameSpaceTest Global 9");
	is(enum_thorough::scientistsNameSpaceTestA($bell), $bell, "scientistsNameSpaceTest Global A");
	is(enum_thorough::scientistsNameSpaceTestB($bell), $bell, "scientistsNameSpaceTest Global B");
	is(enum_thorough::scientistsNameSpaceTestC($bell), $bell, "scientistsNameSpaceTest Global C");
	is(enum_thorough::scientistsNameSpaceTestD($bell), $bell, "scientistsNameSpaceTest Global D");
	is(enum_thorough::scientistsNameSpaceTestE($bell), $bell, "scientistsNameSpaceTest Global E");

	is(enum_thorough::scientistsNameSpaceTestF($bell), $bell, "scientistsNameSpaceTest Global F");
	is(enum_thorough::scientistsNameSpaceTestG($bell), $bell, "scientistsNameSpaceTest Global G");
	is(enum_thorough::scientistsNameSpaceTestH($bell), $bell, "scientistsNameSpaceTest Global H");
	is(enum_thorough::scientistsNameSpaceTestI($bell), $bell, "scientistsNameSpaceTest Global I");
	is(enum_thorough::scientistsNameSpaceTestJ($bell), $bell, "scientistsNameSpaceTest Global J");
	is(enum_thorough::scientistsNameSpaceTestK($bell), $bell, "scientistsNameSpaceTest Global K");
	is(enum_thorough::scientistsNameSpaceTestL($bell), $bell, "scientistsNameSpaceTest Global L");
}
{
	my $val = $enum_thorough::argh;
	is(enum_thorough::renameTest1($val), $val, "renameTest Global 1");
	is(enum_thorough::renameTest2($val), $val, "renameTest Global 2");
}
{
	my $n = enum_thorough::NewNameStruct->new();
	is($n->renameTest1($enum_thorough::NewNameStruct::bang), $enum_thorough::NewNameStruct::bang, "renameTest 1");
	is($n->renameTest2($enum_thorough::NewNameStruct::bang), $enum_thorough::NewNameStruct::bang, "renameTest 2");
	is($n->renameTest3($enum_thorough::NewNameStruct::simple1), $enum_thorough::NewNameStruct::simple1, "renameTest 3");
	is($n->renameTest4($enum_thorough::NewNameStruct::doublename1), $enum_thorough::NewNameStruct::doublename1, "renameTest 4");
	is($n->renameTest5($enum_thorough::NewNameStruct::doublename1), $enum_thorough::NewNameStruct::doublename1, "renameTest 5");
	is($n->renameTest6($enum_thorough::NewNameStruct::singlename1), $enum_thorough::NewNameStruct::singlename1, "renameTest 6");
}
{
	is(enum_thorough::renameTest3($enum_thorough::NewNameStruct::bang), $enum_thorough::NewNameStruct::bang, "renameTest Global 3");
	is(enum_thorough::renameTest4($enum_thorough::NewNameStruct::simple1), $enum_thorough::NewNameStruct::simple1, "renameTest Global 4");
	is(enum_thorough::renameTest5($enum_thorough::NewNameStruct::doublename1), $enum_thorough::NewNameStruct::doublename1, "renameTest Global 5");
	is(enum_thorough::renameTest6($enum_thorough::NewNameStruct::doublename1), $enum_thorough::NewNameStruct::doublename1, "renameTest Global 6");
	is(enum_thorough::renameTest7($enum_thorough::NewNameStruct::singlename1), $enum_thorough::NewNameStruct::singlename1, "renameTest Global 7");
}
{
	my $t = enum_thorough::TreesClass->new();
	my $pine = $enum_thorough::TreesClass::pine;
	is($t->treesTest1($pine), $pine, "treesTest 1");
	is($t->treesTest2($pine), $pine, "treesTest 2");
	is($t->treesTest3($pine), $pine, "treesTest 3");
	is($t->treesTest4($pine), $pine, "treesTest 4");
	is($t->treesTest5($pine), $pine, "treesTest 5");
	is($t->treesTest6($pine), $pine, "treesTest 6");
	is($t->treesTest7($pine), $pine, "treesTest 7");
	is($t->treesTest8($pine), $pine, "treesTest 8");
	is($t->treesTest9($pine), $pine, "treesTest 9");
	is($t->treesTestA($pine), $pine, "treesTest A");
	is($t->treesTestB($pine), $pine, "treesTest B");
	is($t->treesTestC($pine), $pine, "treesTest C");
	is($t->treesTestD($pine), $pine, "treesTest D");
	is($t->treesTestE($pine), $pine, "treesTest E");
	is($t->treesTestF($pine), $pine, "treesTest F");
	is($t->treesTestG($pine), $pine, "treesTest G");
	is($t->treesTestH($pine), $pine, "treesTest H");
	is($t->treesTestI($pine), $pine, "treesTest I");
	is($t->treesTestJ($pine), $pine, "treesTest J");
	is($t->treesTestK($pine), $pine, "treesTest K");
	is($t->treesTestL($pine), $pine, "treesTest L");
	is($t->treesTestM($pine), $pine, "treesTest M");
	is($t->treesTestN($pine), $pine, "treesTest N");
	is($t->treesTestO($pine), $pine, "treesTest O");

	is(enum_thorough::treesTest1($pine), $pine, "treesTest Global 1");
	is(enum_thorough::treesTest2($pine), $pine, "treesTest Global 2");
	is(enum_thorough::treesTest3($pine), $pine, "treesTest Global 3");
	is(enum_thorough::treesTest4($pine), $pine, "treesTest Global 4");
	is(enum_thorough::treesTest5($pine), $pine, "treesTest Global 5");
	is(enum_thorough::treesTest6($pine), $pine, "treesTest Global 6");
	is(enum_thorough::treesTest7($pine), $pine, "treesTest Global 7");
	is(enum_thorough::treesTest8($pine), $pine, "treesTest Global 8");
	is(enum_thorough::treesTest9($pine), $pine, "treesTest Global 9");
	is(enum_thorough::treesTestA($pine), $pine, "treesTest Global A");
	is(enum_thorough::treesTestB($pine), $pine, "treesTest Global B");
	is(enum_thorough::treesTestC($pine), $pine, "treesTest Global C");
	is(enum_thorough::treesTestD($pine), $pine, "treesTest Global D");
	is(enum_thorough::treesTestE($pine), $pine, "treesTest Global E");
	is(enum_thorough::treesTestF($pine), $pine, "treesTest Global F");
	is(enum_thorough::treesTestG($pine), $pine, "treesTest Global G");
	is(enum_thorough::treesTestH($pine), $pine, "treesTest Global H");
	is(enum_thorough::treesTestI($pine), $pine, "treesTest Global I");
	is(enum_thorough::treesTestJ($pine), $pine, "treesTest Global J");
	is(enum_thorough::treesTestK($pine), $pine, "treesTest Global K");
	is(enum_thorough::treesTestL($pine), $pine, "treesTest Global L");
	is(enum_thorough::treesTestM($pine), $pine, "treesTest Global M");
	is(enum_thorough::treesTestO($pine), $pine, "treesTest Global O");
	is(enum_thorough::treesTestP($pine), $pine, "treesTest Global P");
	is(enum_thorough::treesTestQ($pine), $pine, "treesTest Global Q");
	is(enum_thorough::treesTestR($pine), $pine, "treesTest Global R");
}
{
	my $h = enum_thorough::HairStruct->new();
	my $ginger = $enum_thorough::HairStruct::ginger;

	is($h->hairTest1($ginger), $ginger, "hairTest 1");
	is($h->hairTest2($ginger), $ginger, "hairTest 2");
	is($h->hairTest3($ginger), $ginger, "hairTest 3");
	is($h->hairTest4($ginger), $ginger, "hairTest 4");
	is($h->hairTest5($ginger), $ginger, "hairTest 5");
	is($h->hairTest6($ginger), $ginger, "hairTest 6");
	is($h->hairTest7($ginger), $ginger, "hairTest 7");
	is($h->hairTest8($ginger), $ginger, "hairTest 8");
	is($h->hairTest9($ginger), $ginger, "hairTest 9");
	is($h->hairTestA($ginger), $ginger, "hairTest A");
	is($h->hairTestB($ginger), $ginger, "hairTest B");

	my $red = $enum_thorough::red;
	is($h->colourTest1($red), $red, "colourTest HairStruct 1");
	is($h->colourTest2($red), $red, "colourTest HairStruct 2");
	is($h->namedanonTest1($enum_thorough::NamedAnon2), $enum_thorough::NamedAnon2, "namedanonTest HairStruct 1");
{ local $TODO = "shouldn't namespaces drop into a package?";
	ok(defined($enum_thorough::AnonSpace::NamedAnonSpace2), "found enum value");
SKIP: {
	skip "enum value not in expected package", 1 unless defined $enum_thorough::AnonSpace::NamedAnonSpace2;
	is($h->namedanonspaceTest1($enum_thorough::AnonSpace::NamedAnonSpace2), $enum_thorough::AnonSpace::NamedAnonSpace2, "namedanonspaceTest HairStruct 1");
}}
	

	my $fir = $enum_thorough::TreesClass::fir;
	is($h->treesGlobalTest1($fir), $fir, "treesGlobalTest1 HairStruct 1");
	is($h->treesGlobalTest2($fir), $fir, "treesGlobalTest1 HairStruct 2");
	is($h->treesGlobalTest3($fir), $fir, "treesGlobalTest1 HairStruct 3");
	is($h->treesGlobalTest4($fir), $fir, "treesGlobalTest1 HairStruct 4");
}
{
	my $blonde = $enum_thorough::HairStruct::blonde;
	is(enum_thorough::hairTest1($blonde), $blonde, "hairTest Global 1");
	is(enum_thorough::hairTest2($blonde), $blonde, "hairTest Global 2");
	is(enum_thorough::hairTest3($blonde), $blonde, "hairTest Global 3");
	is(enum_thorough::hairTest4($blonde), $blonde, "hairTest Global 4");
	is(enum_thorough::hairTest5($blonde), $blonde, "hairTest Global 5");
	is(enum_thorough::hairTest6($blonde), $blonde, "hairTest Global 6");
	is(enum_thorough::hairTest7($blonde), $blonde, "hairTest Global 7");
	is(enum_thorough::hairTest8($blonde), $blonde, "hairTest Global 8");
	is(enum_thorough::hairTest9($blonde), $blonde, "hairTest Global 9");
	is(enum_thorough::hairTestA($blonde), $blonde, "hairTest Global A");
	is(enum_thorough::hairTestB($blonde), $blonde, "hairTest Global B");
	is(enum_thorough::hairTestC($blonde), $blonde, "hairTest Global C");

	is(enum_thorough::hairTestA1($blonde), $blonde, "hairTest Global A1");
	is(enum_thorough::hairTestA2($blonde), $blonde, "hairTest Global A2");
	is(enum_thorough::hairTestA3($blonde), $blonde, "hairTest Global A3");
	is(enum_thorough::hairTestA4($blonde), $blonde, "hairTest Global A4");
	is(enum_thorough::hairTestA5($blonde), $blonde, "hairTest Global A5");
	is(enum_thorough::hairTestA6($blonde), $blonde, "hairTest Global A6");
	is(enum_thorough::hairTestA7($blonde), $blonde, "hairTest Global A7");
	is(enum_thorough::hairTestA8($blonde), $blonde, "hairTest Global A8");
	is(enum_thorough::hairTestA9($blonde), $blonde, "hairTest Global A9");
	is(enum_thorough::hairTestAA($blonde), $blonde, "hairTest Global AA");
	is(enum_thorough::hairTestAB($blonde), $blonde, "hairTest Global AB");
	is(enum_thorough::hairTestAC($blonde), $blonde, "hairTest Global AC");

	is(enum_thorough::hairTestB1($blonde), $blonde, "hairTest Global B1");
	is(enum_thorough::hairTestB2($blonde), $blonde, "hairTest Global B2");
	is(enum_thorough::hairTestB3($blonde), $blonde, "hairTest Global B3");
	is(enum_thorough::hairTestB4($blonde), $blonde, "hairTest Global B4");
	is(enum_thorough::hairTestB5($blonde), $blonde, "hairTest Global B5");
	is(enum_thorough::hairTestB6($blonde), $blonde, "hairTest Global B6");
	is(enum_thorough::hairTestB7($blonde), $blonde, "hairTest Global B7");
	is(enum_thorough::hairTestB8($blonde), $blonde, "hairTest Global B8");
	is(enum_thorough::hairTestB9($blonde), $blonde, "hairTest Global B9");
	is(enum_thorough::hairTestBA($blonde), $blonde, "hairTest Global BA");
	is(enum_thorough::hairTestBB($blonde), $blonde, "hairTest Global BB");
	is(enum_thorough::hairTestBC($blonde), $blonde, "hairTest Global BC");

	is(enum_thorough::hairTestC1($blonde), $blonde, "hairTest Global C1");
	is(enum_thorough::hairTestC2($blonde), $blonde, "hairTest Global C2");
	is(enum_thorough::hairTestC3($blonde), $blonde, "hairTest Global C3");
	is(enum_thorough::hairTestC4($blonde), $blonde, "hairTest Global C4");
	is(enum_thorough::hairTestC5($blonde), $blonde, "hairTest Global C5");
	is(enum_thorough::hairTestC6($blonde), $blonde, "hairTest Global C6");
	is(enum_thorough::hairTestC7($blonde), $blonde, "hairTest Global C7");
	is(enum_thorough::hairTestC8($blonde), $blonde, "hairTest Global C8");
	is(enum_thorough::hairTestC9($blonde), $blonde, "hairTest Global C9");
	is(enum_thorough::hairTestCA($blonde), $blonde, "hairTest Global CA");
	is(enum_thorough::hairTestCB($blonde), $blonde, "hairTest Global CB");
	is(enum_thorough::hairTestCC($blonde), $blonde, "hairTest Global CC");
}
{
	my $f = enum_thorough::FirStruct->new();
	my $blonde = $enum_thorough::HairStruct::blonde;

	is($f->hairTestFir1($blonde), $blonde, "hairTestFir 1");
	is($f->hairTestFir2($blonde), $blonde, "hairTestFir 2");
	is($f->hairTestFir3($blonde), $blonde, "hairTestFir 3");
	is($f->hairTestFir4($blonde), $blonde, "hairTestFir 4");
	is($f->hairTestFir5($blonde), $blonde, "hairTestFir 5");
	is($f->hairTestFir6($blonde), $blonde, "hairTestFir 6");
	is($f->hairTestFir7($blonde), $blonde, "hairTestFir 7");
	is($f->hairTestFir8($blonde), $blonde, "hairTestFir 8");
	is($f->hairTestFir9($blonde), $blonde, "hairTestFir 9");
	is($f->hairTestFirA($blonde), $blonde, "hairTestFir A");
}
{
	$enum_thorough::GlobalInstance = $enum_thorough::globalinstance2;
	is($enum_thorough::GlobalInstance, $enum_thorough::globalinstance2, "GlobalInstance 1");

	my $i = enum_thorough::Instances->new();
	$i->{MemberInstance} = $enum_thorough::Instances::memberinstance3;
	is($i->{MemberInstance}, $enum_thorough::Instances::memberinstance3, "MemberInstance 1");
}
# ignore enum item tests start
{
	is(enum_thorough::ignoreATest($enum_thorough::IgnoreTest::ignoreA_zero), 0, "ignoreATest 0");
	is(enum_thorough::ignoreATest($enum_thorough::IgnoreTest::ignoreA_three), 3, "ignoreATest 3");
	is(enum_thorough::ignoreATest($enum_thorough::IgnoreTest::ignoreA_ten), 10, "ignoreATest 10");
	is(enum_thorough::ignoreATest($enum_thorough::IgnoreTest::ignoreA_eleven), 11, "ignoreATest 11");
	is(enum_thorough::ignoreATest($enum_thorough::IgnoreTest::ignoreA_thirteen), 13, "ignoreATest 13");
	is(enum_thorough::ignoreATest($enum_thorough::IgnoreTest::ignoreA_fourteen), 14, "ignoreATest 14");
	is(enum_thorough::ignoreATest($enum_thorough::IgnoreTest::ignoreA_twenty), 20, "ignoreATest 20");
	is(enum_thorough::ignoreATest($enum_thorough::IgnoreTest::ignoreA_thirty), 30, "ignoreATest 30");
	is(enum_thorough::ignoreATest($enum_thorough::IgnoreTest::ignoreA_thirty_two), 32, "ignoreATest 32");
	is(enum_thorough::ignoreATest($enum_thorough::IgnoreTest::ignoreA_thirty_three), 33, "ignoreATest 33");
}
{
	is(enum_thorough::ignoreBTest($enum_thorough::IgnoreTest::ignoreB_eleven), 11, "ignoreBTest 11");
	is(enum_thorough::ignoreBTest($enum_thorough::IgnoreTest::ignoreB_twelve), 12, "ignoreBTest 12");
	is(enum_thorough::ignoreBTest($enum_thorough::IgnoreTest::ignoreB_thirty_one), 31, "ignoreBTest 31");
	is(enum_thorough::ignoreBTest($enum_thorough::IgnoreTest::ignoreB_thirty_two), 32, "ignoreBTest 32");
	is(enum_thorough::ignoreBTest($enum_thorough::IgnoreTest::ignoreB_forty_one), 41, "ignoreBTest 41");
	is(enum_thorough::ignoreBTest($enum_thorough::IgnoreTest::ignoreB_forty_two), 42, "ignoreBTest 42");
}
{
	is(enum_thorough::ignoreCTest($enum_thorough::IgnoreTest::ignoreC_ten), 10, "ignoreCTest 10");
	is(enum_thorough::ignoreCTest($enum_thorough::IgnoreTest::ignoreC_twelve), 12, "ignoreCTest 12");
	is(enum_thorough::ignoreCTest($enum_thorough::IgnoreTest::ignoreC_thirty), 30, "ignoreCTest 30");
	is(enum_thorough::ignoreCTest($enum_thorough::IgnoreTest::ignoreC_thirty_two), 32, "ignoreCTest 32");
	is(enum_thorough::ignoreCTest($enum_thorough::IgnoreTest::ignoreC_forty), 40, "ignoreCTest 40");
	is(enum_thorough::ignoreCTest($enum_thorough::IgnoreTest::ignoreC_forty_two), 42, "ignoreCTest 42");
}
{
	is(enum_thorough::ignoreDTest($enum_thorough::IgnoreTest::ignoreD_twenty_one), 21, "ignoreDTest 21");
	is(enum_thorough::ignoreDTest($enum_thorough::IgnoreTest::ignoreD_twenty_two), 22, "ignoreDTest 22");
}
{
	is(enum_thorough::ignoreETest($enum_thorough::IgnoreTest::ignoreE_zero), 0, "ignoreETest 0");
	is(enum_thorough::ignoreETest($enum_thorough::IgnoreTest::ignoreE_twenty_one), 21, "ignoreETest 21");
	is(enum_thorough::ignoreETest($enum_thorough::IgnoreTest::ignoreE_twenty_two), 22, "ignoreETest 22");
}
# ignore enum item tests end
{
	is(enum_thorough::repeatTest($enum_thorough::one), 1, "repeatTest 1");
	is(enum_thorough::repeatTest($enum_thorough::initial), 1, "repeatTest 2");
	is(enum_thorough::repeatTest($enum_thorough::two), 2, "repeatTest 3");
	is(enum_thorough::repeatTest($enum_thorough::three), 3, "repeatTest 4");
	is(enum_thorough::repeatTest($enum_thorough::llast), 3, "repeatTest 5");
	is(enum_thorough::repeatTest($enum_thorough::end), 3, "repeatTest 6");
}

# these were the preexisting Perl testcases before the port.

# Just test an in and out typemap for enum SWIGTYPE and const enum SWIGTYPE & typemaps
is(enum_thorough::speedTest4($enum_thorough::SpeedClass::slow),
	$enum_thorough::SpeedClass::slow, "speedTest Global 4");
is(enum_thorough::speedTest5($enum_thorough::SpeedClass::slow),
	$enum_thorough::SpeedClass::slow, "speedTest Global 5");
is(enum_thorough::speedTest4($enum_thorough::SpeedClass::fast),
	$enum_thorough::SpeedClass::fast, "speedTest Global 4");
is(enum_thorough::speedTest5($enum_thorough::SpeedClass::fast),
	$enum_thorough::SpeedClass::fast, "speedTest Global 5");
